import functools
import importlib
import json
import os
import tarfile
from typing import List, Tuple
import zipfile
from collections import Callable
from ruamel import yaml
import requests
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch import Tensor
import argparse
class InputData():
    """input datas class
    """
    def __init__(self, inputs: List =None):
        """init input datas class
        
        if inputs is None:
            this class can be used to save all InputData in the history by 'merge_input_data(X:InputData)' 
        else:
            this class can be used for model input.
        
        Args:
            inputs (List, optional): inputs with [tokenized_data, slot, intent]. Defaults to None.
        """
        if inputs == None:
            self.slot = []
            self.intent = []
            self.input_ids = None
            self.token_type_ids = None
            self.attention_mask = None
            self.seq_lens = None
        else:
            self.input_ids = inputs[0].input_ids
            self.token_type_ids = None
            if hasattr(inputs[0], "token_type_ids"):
                self.token_type_ids = inputs[0].token_type_ids
            self.attention_mask = inputs[0].attention_mask
            if len(inputs)>=2:
                self.slot = inputs[1]
            if len(inputs)>=3:
                self.intent = inputs[2]
            self.seq_lens = self.attention_mask.sum(-1)

    def get_inputs(self):
        """ get tokenized_data

        Returns:
            dict: tokenized data
        """
        res = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask
        }
        if self.token_type_ids is not None:
            res["token_type_ids"] = self.token_type_ids
        return res

    def merge_input_data(self, inp: "InputData"):
        """merge another InputData object with slot and intent

        Args:
            inp (InputData): another InputData object
        """
        self.slot += inp.slot
        self.intent += inp.intent

    def get_slot_mask(self, ignore_index:int)->Tensor:
        """get slot mask 

        Args:
            ignore_index (int): ignore index used in slot padding
        
        Returns:
            Tensor: mask tensor
        """
        mask = self.slot != ignore_index
        mask[:, 0] = torch.ones_like(mask[:, 0]).to(self.slot.device)
        return mask

    def get_item(self, index, tokenizer=None, intent_map=None, slot_map=None, ignore_index = -100, use_multi=False):
        res = {"input_ids": self.input_ids[index]}
        if tokenizer is not None:
            res["tokens"] = [tokenizer.decode(x) for x in self.input_ids[index]]
        if intent_map is not None:
            intents = self.intent.tolist()
            if isinstance(intents[index], list):
                if use_multi:
                    res["intent"] = [intent_map[i] for i, x in enumerate(intents[index]) if x == 1.]
                else:
                    res["intent"] = [intent_map[int(x)] for x in intents[index]]
            else:
                res["intent"] = intent_map[intents[index]]
        if slot_map is not None:
            res["slot"] = [slot_map[x] if x != ignore_index else "#" for x in self.slot.tolist()[index]]
        return res

class OutputData():
    """output data class
    """
    def __init__(self, intent_ids=None, slot_ids=None):
        """init output data class
        
        if intent_ids is None and slot_ids is None:
            this class can be used to save all OutputData in the history by 'merge_output_data(X:OutputData)' 
        else:
            this class can be used to model output management.
        
        Args:
            intent_ids (Any, optional): list(Tensor) of intent ids / logits / strings. Defaults to None.
            slot_ids (Any, optional): list(Tensor) of slot ids / ids / strings. Defaults to None.
        """
        if intent_ids is None and slot_ids is None:
            self.intent_ids = []
            self.slot_ids = []
        else:
            if isinstance(intent_ids, ClassifierOutputData):
                self.intent_ids = intent_ids.classifier_output
            else:
                self.intent_ids = intent_ids
            if isinstance(slot_ids, ClassifierOutputData):
                self.slot_ids = slot_ids.classifier_output
            else:
                self.slot_ids = slot_ids

    def map_output(self, slot_map=None, intent_map=None):
        """ map intent or slot ids to intent or slot string.

        Args:
            slot_map (dict, optional): slot id-to-string map. Defaults to None.
            intent_map (dict, optional): intent id-to-string map. Defaults to None.
        """
        if self.slot_ids is not None:
            if slot_map:
                self.slot_ids = [[slot_map[x] if x >= 0 else "#" for x in sid] for sid in self.slot_ids]
        if self.intent_ids is not None:
            if intent_map:
                self.intent_ids = [[intent_map[x] for x in sid] if isinstance(sid, list) else intent_map[sid] for sid in
                                   self.intent_ids]

    def merge_output_data(self, output:"OutputData"):
        """merge another OutData object with slot and intent

        Args:
            output (OutputData): another OutputData object
        """
        if output.slot_ids is not None:
            self.slot_ids += output.slot_ids
        if output.intent_ids is not None:
            self.intent_ids += output.intent_ids

    def save(self, path:str, original_dataset=None):
        """ save all OutputData in the history

        Args:
            path (str): save dir path
            original_dataset(Iterable): original dataset
        """
        # with open(f"{path}/intent.jsonl", "w") as f:
        #     for x in self.intent_ids:
        #         f.write(json.dumps(x) + "\n")
        with open(f"{path}/outputs.jsonl", "w") as f:
            if original_dataset is not None:
                for i, s, d in zip(self.intent_ids, self.slot_ids, original_dataset):
                    f.write(json.dumps({"pred_intent": i, "pred_slot": s, "text": d["text"], "golden_intent":d["intent"], "golden_slot":d["slot"]}, ensure_ascii=False) + "\n")
            else:
                for i, s in zip(self.intent_ids, self.slot_ids):
                    f.write(json.dumps({"pred_intent": i, "pred_slot": s}, ensure_ascii=False) + "\n")


class HiddenData():
    """Interactive data structure for all model components
    """
    def __init__(self, intent_hidden, slot_hidden):
        """init hidden data structure

        Args:
            intent_hidden (Any): sentence-level or intent hidden state
            slot_hidden (Any): token-level or slot hidden state
        """
        self.intent_hidden = intent_hidden
        self.slot_hidden = slot_hidden
        self.inputs = None
        self.embedding = None

    def get_intent_hidden_state(self):
        """get intent hidden state

        Returns:
            Any: intent hidden state
        """
        return self.intent_hidden

    def get_slot_hidden_state(self):
        """get slot hidden state

        Returns:
            Any: slot hidden state
        """
        return self.slot_hidden

    def update_slot_hidden_state(self, hidden_state):
        """update slot hidden state

        Args:
            hidden_state (Any): slot hidden state to update
        """
        self.slot_hidden = hidden_state

    def update_intent_hidden_state(self, hidden_state):
        """update intent hidden state

        Args:
            hidden_state (Any): intent hidden state to update
        """
        self.intent_hidden = hidden_state

    def add_input(self, inputs: InputData or "HiddenData"):
        """add last model component input information to next model component

        Args:
            inputs (InputDataor or HiddenData): last model component input
        """
        self.inputs = inputs

    def add_embedding(self, embedding):
        self.embedding = embedding


class ClassifierOutputData():
    """Classifier output data structure of all classifier components
    """
    def __init__(self, classifier_output):
        self.classifier_output = classifier_output
        self.output_embedding = None

def remove_slot_ignore_index(inputs:InputData, outputs:OutputData, ignore_index=-100):
    """ remove padding or extra token in input id and output id

    Args:
        inputs (InputData): input data with input id
        outputs (OutputData): output data with decoded output id
        ignore_index (int, optional): ignore_index in input_ids. Defaults to -100.

    Returns:
        InputData: input data removed padding or extra token
        OutputData: output data removed padding or extra token
    """
    for index, (inp_ss, out_ss) in enumerate(zip(inputs.slot, outputs.slot_ids)):
        temp_inp = []
        temp_out = []
        for inp_s, out_s in zip(list(inp_ss), list(out_ss)):
            if inp_s != ignore_index:
                temp_inp.append(inp_s)
                temp_out.append(out_s)

        inputs.slot[index] = temp_inp
        outputs.slot_ids[index] = temp_out
    return inputs, outputs


def pack_sequence(inputs:Tensor, seq_len:Tensor or List) -> Tensor:
    """pack sequence data to packed data without padding.
    
    Args:
        inputs (Tensor): list(Tensor) of packed sequence inputs
        seq_len (Tensor or List): list(Tensor) of sequence length

    Returns:
        Tensor: packed inputs
        
    Examples:
        inputs = [[x, y, z, PAD, PAD], [x, y, PAD, PAD, PAD]]
        
        seq_len = [3,2]
        
        return -> [x, y, z, x, y]
    """
    output = []
    for index, batch in enumerate(inputs):
        output.append(batch[:seq_len[index]])
    return torch.cat(output, dim=0)


def unpack_sequence(inputs:Tensor, seq_lens:Tensor or List, padding_value=0) -> Tensor:
    """unpack sequence data.
        
    Args:
        inputs (Tensor): list(Tensor) of packed sequence inputs
        seq_lens (Tensor or List): list(Tensor) of sequence length
        padding_value (int, optional): padding value. Defaults to 0.

    Returns:
        Tensor: unpacked inputs
        
    Examples:
        inputs = [x, y, z, x, y]
        
        seq_len = [3,2]
        
        return -> [[x, y, z, PAD, PAD], [x, y, PAD, PAD, PAD]]
    """
    last_idx = 0
    output = []
    for _, seq_len in enumerate(seq_lens):
        output.append(inputs[last_idx:last_idx + seq_len])
        last_idx = last_idx + seq_len
    return pad_sequence(output, batch_first=True, padding_value=padding_value)


def get_dict_with_key_prefix(input_dict: dict, prefix=""):
    res = {}
    for t in input_dict:
        res[t + prefix] = input_dict[t]
    return res


def download(url: str, fname: str):
    """download file from url to fname

    Args:
        url (str): remote server url path
        fname (str): local path to save
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def tar_gz_data(file_name:str):
    """use "tar.gz" format to compress data

    Args:
        file_name (str): file path to tar
    """
    t = tarfile.open(f"{file_name}.tar.gz", "w:gz")

    for root, dir, files in os.walk(f"{file_name}"):
        print(root, dir, files)
        for file in files:
            fullpath = os.path.join(root, file)
            t.add(fullpath)
    t.close()


def untar(fname:str, dirs:str):
    """ uncompress "tar.gz" file

    Args:
        fname (str): file path to untar
        dirs (str): target dir path
    """
    t = tarfile.open(fname)
    t.extractall(path=dirs)


def unzip_file(zip_src:str, dst_dir:str):
    """ uncompress "zip" file

    Args:
        fname (str): file path to unzip
        dirs (str): target dir path
    """
    r = zipfile.is_zipfile(zip_src)
    if r:
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')


def find_callable(target: str) -> Callable:
    """ find callable function / class to instantiate

    Args:
        target (str): class/module path

    Raises:
        e: can not import module

    Returns:
        Callable: return function / class
    """
    target_module_path, target_callable_path = target.rsplit(".", 1)
    target_callable_paths = [target_callable_path]

    target_module = None
    while len(target_module_path):
        try:
            target_module = importlib.import_module(target_module_path)
            break
        except Exception as e:
            raise e
    target_callable = target_module
    for attr in reversed(target_callable_paths):
        target_callable = getattr(target_callable, attr)

    return target_callable


def instantiate(config, target="_model_target_", partial="_model_partial_"):
    """ instantiate object by config.
    
    Modified from https://github.com/HIT-SCIR/ltp/blob/main/python/core/ltp_core/models/utils/instantiate.py.

    Args:
        config (Any): configuration
        target (str, optional): key to assign the class to be instantiated. Defaults to "_model_target_".
        partial (str, optional): key to judge object whether should be instantiated partially. Defaults to "_model_partial_".

    Returns:
        Any: instantiated object
    """
    if isinstance(config, dict) and target in config:
        target_path = config.get(target)
        target_callable = find_callable(target_path)

        is_partial = config.get(partial, False)
        target_args = {
            key: instantiate(value)
            for key, value in config.items()
            if key not in [target, partial]
        }

        if is_partial:
            return functools.partial(target_callable, **target_args)
        else:
            return target_callable(**target_args)
    elif isinstance(config, dict):
        return {key: instantiate(value) for key, value in config.items()}
    else:
        return config


def load_yaml(file):
    """ load data from yaml files.

    Args:
        file (str): yaml file path.

    Returns:
        Any: data
    """
    with open(file, encoding="utf-8") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc

def from_configured(configure_name_or_file:str, model_class:Callable, config_prefix="./config/", **input_config):
    """load module from pre-configured data

    Args:
        configure_name_or_file (str): config path -> {config_prefix}/{configure_name_or_file}.yaml
        model_class (Callable): module class
        config_prefix (str, optional): configuration root path. Defaults to "./config/".

    Returns:
        Any: instantiated object.
    """
    if os.path.exists(configure_name_or_file):
        configure_file=configure_name_or_file
    else:
        configure_file= os.path.join(config_prefix, configure_name_or_file+".yaml")
    config = load_yaml(configure_file)
    config.update(input_config)
    return model_class(**config)

def save_json(file_path, obj):
    with open(file_path, 'w', encoding="utf8") as fw:
            fw.write(json.dumps(obj))
            
def load_json(file_path):
    with open(file_path, 'r', encoding="utf8") as fw:
        res =json.load(fw)
    return res

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')