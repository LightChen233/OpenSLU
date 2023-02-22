'''
Author: Qiguang Chen
Date: 2023-01-11 10:39:26
LastEditors: Qiguang Chen
LastEditTime: 2023-02-19 15:39:48
Description: all class for load data.

'''
import os
import torch
import json
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader

from common.utils import InputData

ABS_PATH=os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")

class DataFactory(object):
    def __init__(self, tokenizer,use_multi_intent=False, to_lower_case=True):
        """_summary_

        Args:
            tokenizer (Tokenizer): _description_
            use_multi_intent (bool, optional): _description_. Defaults to False.
        """
        self.tokenizer = tokenizer
        self.slot_label_list = []
        self.intent_label_list = []
        self.use_multi = use_multi_intent
        self.to_lower_case = to_lower_case
        self.slot_label_dict = None
        self.intent_label_dict = None

    def __is_supported_datasets(self, dataset_name:str)->bool:
        return dataset_name.lower() in ["atis", "snips", "mix-atis", "mix-atis"]

    def load_dataset(self, dataset_config, split="train"):
        dataset_name = None
        if split not in dataset_config:
            dataset_name = dataset_config.get("dataset_name")
        elif self.__is_supported_datasets(dataset_config[split]):
            dataset_name = dataset_config[split].lower()
        if dataset_name is not None:
            return load_dataset("LightChen2333/OpenSLU", dataset_name, split=split)
        else:
            data_file = dataset_config[split]
            data_dict = {"text": [], "slot": [], "intent":[]}
            with open(data_file, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    data_dict["text"].append(row["text"])
                    data_dict["slot"].append(row["slot"])
                    data_dict["intent"].append(row["intent"])
            return Dataset.from_dict(data_dict)

    def update_label_names(self, dataset):
        for intent_labels in dataset["intent"]:
            if self.use_multi:
                intent_label = intent_labels.split("#")
            else:
                intent_label = [intent_labels]
            for x in intent_label:
                if x not in self.intent_label_list:
                    self.intent_label_list.append(x)
        for slot_label in dataset["slot"]:
            for x in slot_label:
                if x not in self.slot_label_list:
                    self.slot_label_list.append(x)
        self.intent_label_dict = {key: index for index,
                                  key in enumerate(self.intent_label_list)}
        self.slot_label_dict = {key: index for index,
                                key in enumerate(self.slot_label_list)}

    def update_vocabulary(self, dataset):
        if self.tokenizer.name_or_path in ["word_tokenizer"]:
            for data in dataset:
                self.tokenizer.add_instance(data["text"])

    @staticmethod
    def fast_align_data(text, padding_side="right"):
        for i in range(len(text.input_ids)):
            desired_output = []
            for word_id in text.word_ids(i):
                if word_id is not None:
                    start, end = text.word_to_tokens(
                        i, word_id, sequence_index=0 if padding_side == "right" else 1)
                    if start == end - 1:
                        tokens = [start]
                    else:
                        tokens = [start, end - 1]
                    if len(desired_output) == 0 or desired_output[-1] != tokens:
                        desired_output.append(tokens)
            yield desired_output

    def fast_align(self,
                   batch,
                   ignore_index=-100,
                   device="cuda",
                   config=None,
                   enable_label=True,
                   label2tensor=True):
        if self.to_lower_case:
            input_list = [[t.lower() for t in x["text"]] for x in batch]
        else:
            input_list = [x["text"] for x in batch]
        text = self.tokenizer(input_list,
                              return_tensors="pt",
                              padding=True,
                              is_split_into_words=True,
                              truncation=True,
                              **config).to(device)
        if enable_label:
            if label2tensor:

                slot_mask = torch.ones_like(text.input_ids) * ignore_index
                for i, offsets in enumerate(
                        DataFactory.fast_align_data(text, padding_side=self.tokenizer.padding_side)):
                    num = 0
                    assert len(offsets) == len(batch[i]["text"])
                    assert len(offsets) == len(batch[i]["slot"])
                    for off in offsets:
                        slot_mask[i][off[0]
                                     ] = self.slot_label_dict[batch[i]["slot"][num]]
                        num += 1
                slot = slot_mask.clone()
                attentin_id = 0 if self.tokenizer.padding_side == "right" else 1
                for i, slot_batch in enumerate(slot):
                    for j, x in enumerate(slot_batch):
                        if x == ignore_index and text.attention_mask[i][j] == attentin_id and (text.input_ids[i][
                                j] not in self.tokenizer.all_special_ids or text.input_ids[i][j] == self.tokenizer.unk_token_id):
                            slot[i][j] = slot[i][j - 1]
                slot = slot.to(device)
                if not self.use_multi:
                    intent = torch.tensor(
                        [self.intent_label_dict[x["intent"]] for x in batch]).to(device)
                else:
                    one_hot = torch.zeros(
                        (len(batch), len(self.intent_label_list)), dtype=torch.float)
                    for index, b in enumerate(batch):
                        for x in b["intent"].split("#"):
                            one_hot[index][self.intent_label_dict[x]] = 1.
                    intent = one_hot.to(device)
            else:
                slot_mask = None
                slot = [['#' for _ in range(text.input_ids.shape[1])]
                        for _ in range(text.input_ids.shape[0])]
                for i, offsets in enumerate(DataFactory.fast_align_data(text)):
                    num = 0
                    for off in offsets:
                        slot[i][off[0]] = batch[i]["slot"][num]
                        num += 1
                if not self.use_multi:
                    intent = [x["intent"] for x in batch]
                else:
                    intent = [
                        [x for x in b["intent"].split("#")] for b in batch]
            return InputData((text, slot, intent))
        else:
            return InputData((text, None, None))

    def general_align_data(self, split_text_list, raw_text_list, encoded_text):
        for i in range(len(split_text_list)):
            desired_output = []
            jdx = 0
            offset = encoded_text.offset_mapping[i].tolist()
            split_texts = split_text_list[i]
            raw_text = raw_text_list[i]
            last = 0
            temp_offset = []
            for off in offset:
                s, e = off
                if len(temp_offset) > 0 and (e != 0 and last == s):
                    len_1 = off[1] - off[0]
                    len_2 = temp_offset[-1][1] - temp_offset[-1][0]
                    if len_1 > len_2:
                        temp_offset.pop(-1)
                        temp_offset.append([0, 0])
                        temp_offset.append(off)
                    continue
                temp_offset.append(off)
                last = s
            offset = temp_offset
            for split_text in split_texts:
                while jdx < len(offset) and offset[jdx][0] == 0 and offset[jdx][1] == 0:
                    jdx += 1
                if jdx == len(offset):
                    continue
                start_, end_ = offset[jdx]
                tokens = None
                if split_text == raw_text[start_:end_].strip():
                    tokens = [jdx]
                else:
                    # Compute "xxx" -> "xx" "#x"
                    temp_jdx = jdx
                    last_str = raw_text[start_:end_].strip()
                    while last_str != split_text and temp_jdx < len(offset) - 1:
                        temp_jdx += 1
                        last_str += raw_text[offset[temp_jdx]
                                             [0]:offset[temp_jdx][1]].strip()

                    if temp_jdx == jdx:
                        raise ValueError("Illegal Input data")
                    elif last_str == split_text:
                        tokens = [jdx, temp_jdx]
                        jdx = temp_jdx
                    else:
                        jdx -= 1
                jdx += 1
                if tokens is not None:
                    desired_output.append(tokens)
            yield desired_output

    def general_align(self,
                      batch,
                      ignore_index=-100,
                      device="cuda",
                      config=None,
                      enable_label=True,
                      label2tensor=True,
                      locale="en-US"):
        if self.to_lower_case:
            raw_data = [" ".join(x["text"]).lower() if locale not in ['ja-JP', 'zh-CN', 'zh-TW'] else "".join(x["text"]) for x in
                    batch]
            input_list = [[t.lower() for t in x["text"]] for x in batch]
        else:
            input_list = [x["text"] for x in batch]
            raw_data = [" ".join(x["text"]) if locale not in ['ja-JP', 'zh-CN', 'zh-TW'] else "".join(x["text"]) for x in
                        batch]
        text = self.tokenizer(raw_data,
                              return_tensors="pt",
                              padding=True,
                              truncation=True,
                              return_offsets_mapping=True,
                              **config).to(device)
        if enable_label:
            if label2tensor:
                slot_mask = torch.ones_like(text.input_ids) * ignore_index
                for i, offsets in enumerate(
                        self.general_align_data(input_list, raw_data, encoded_text=text)):
                    num = 0
                    # if len(offsets) != len(batch[i]["text"]) or len(offsets) != len(batch[i]["slot"]):
                    #     if
                    for off in offsets:
                        slot_mask[i][off[0]
                                     ] = self.slot_label_dict[batch[i]["slot"][num]]
                        num += 1
                # slot = slot_mask.clone()
                # attentin_id = 0 if self.tokenizer.padding_side == "right" else 1
                # for i, slot_batch in enumerate(slot):
                #     for j, x in enumerate(slot_batch):
                #         if x == ignore_index and text.attention_mask[i][j] == attentin_id and text.input_ids[i][
                #             j] not in self.tokenizer.all_special_ids:
                #             slot[i][j] = slot[i][j - 1]
                slot = slot_mask.to(device)
                if not self.use_multi:
                    intent = torch.tensor(
                        [self.intent_label_dict[x["intent"]] for x in batch]).to(device)
                else:
                    one_hot = torch.zeros(
                        (len(batch), len(self.intent_label_list)), dtype=torch.float)
                    for index, b in enumerate(batch):
                        for x in b["intent"].split("#"):
                            one_hot[index][self.intent_label_dict[x]] = 1.
                    intent = one_hot.to(device)
            else:
                slot_mask = None
                slot = [['#' for _ in range(text.input_ids.shape[1])]
                        for _ in range(text.input_ids.shape[0])]
                for i, offsets in enumerate(self.general_align_data(input_list, raw_data, encoded_text=text)):
                    num = 0
                    for off in offsets:
                        slot[i][off[0]] = batch[i]["slot"][num]
                        num += 1
                if not self.use_multi:
                    intent = [x["intent"] for x in batch]
                else:
                    intent = [
                        [x for x in b["intent"].split("#")] for b in batch]
            return InputData((text, slot, intent))
        else:
            return InputData((text, None, None))

    def batch_fn(self,
                 batch,
                 ignore_index=-100,
                 device="cuda",
                 config=None,
                 align_mode="fast",
                 enable_label=True,
                 label2tensor=True):
        if align_mode == "fast":
            # try:
            return self.fast_align(batch,
                                   ignore_index=ignore_index,
                                   device=device,
                                   config=config,
                                   enable_label=enable_label,
                                   label2tensor=label2tensor)
            # except:
            #     return self.general_align(batch,
            #                               ignore_index=ignore_index,
            #                               device=device,
            #                               config=config,
            #                               enable_label=enable_label,
            #                               label2tensor=label2tensor)
        else:
            return self.general_align(batch,
                                      ignore_index=ignore_index,
                                      device=device,
                                      config=config,
                                      enable_label=enable_label,
                                      label2tensor=label2tensor)

    def get_data_loader(self,
                        dataset,
                        batch_size,
                        shuffle=False,
                        device="cuda",
                        enable_label=True,
                        align_mode="fast",
                        label2tensor=True, **config):
        data_loader = DataLoader(dataset,
                                 shuffle=shuffle,
                                 batch_size=batch_size,
                                 collate_fn=lambda x: self.batch_fn(x,
                                                                    device=device,
                                                                    config=config,
                                                                    enable_label=enable_label,
                                                                    align_mode=align_mode,
                                                                    label2tensor=label2tensor))
        return data_loader
