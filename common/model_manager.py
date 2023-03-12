'''
Author: Qiguang Chen
Date: 2023-01-11 10:39:26
LastEditors: Qiguang Chen
LastEditTime: 2023-03-07 23:49:19
Description: manage all process of model training and prediction.

'''
import math
import os
import queue
import random

import numpy as np
import torch
from tqdm import tqdm


from common import utils
from common.loader import DataFactory
from common.logger import Logger
from common.metric import Evaluator
from common.saver import Saver
from common.tokenizer import get_tokenizer, get_tokenizer_class, load_embedding
from common.utils import InputData, instantiate
from common.utils import OutputData
from common.config import Config
import dill
from common import global_pool
from tools.load_from_hugging_face import PreTrainedTokenizerForSLU, PretrainedModelForSLU
# from tools.hugging_face_parser import load_model, load_tokenizer


class ModelManager(object):
    def __init__(self, config: Config):
        """create model manager by config

        Args:
            config (Config): configuration to manage all process in OpenSLU
        """
        # init config
        global_pool._init()
        self.config = config
        self.__set_seed(self.config.base.get("seed"))
        self.device = self.config.base.get("device")
        self.load_dir = self.config.model_manager.get("load_dir")
        if self.config.get("logger") and self.config["logger"].get("logger_type"):
            logger_type = self.config["logger"].get("logger_type")
        else:
            logger_type = "wandb"
        # enable accelerator
        if "accelerator" in self.config and self.config["accelerator"].get("use_accelerator"):
            from accelerate import Accelerator
            self.accelerator = Accelerator(log_with=logger_type)
        else:
            self.accelerator = None
        self.tokenizer = None
        self.saver = Saver(self.config.model_manager, start_time=self.config.start_time, accelerator=self.accelerator)
        if self.config.base.get("train"):
            self.model = None
            self.optimizer = None
            self.total_step = None
            self.lr_scheduler = None
        self.init_step = 0
        self.best_metric = 0
        self.logger = Logger(logger_type=logger_type,
                             logger_name=self.config.base["name"],
                             start_time=self.config.start_time,
                             accelerator=self.accelerator)
        global_pool.set_value("logger", self.logger)
        
    def init_model(self):
        """init model, optimizer, lr_scheduler

        Args:
            model (Any): pytorch model
        """
        self.prepared = False
        if self.load_dir is not None:
            self.load()
            self.config.set_vocab_size(self.tokenizer.vocab_size)
            self.init_data()
            if self.config.base.get("train") and self.config.model_manager.get("load_train_state"):
                train_state = torch.load(os.path.join(
                    self.load_dir, "train_state.pkl"), pickle_module=dill)
                self.optimizer = instantiate(
                    self.config["optimizer"])(self.model.parameters())
                self.lr_scheduler = instantiate(self.config["scheduler"])(
                    optimizer=self.optimizer,
                    num_training_steps=self.total_step
                )
                self.optimizer.load_state_dict(train_state["optimizer"])
                self.optimizer.zero_grad()
                self.lr_scheduler.load_state_dict(train_state["lr_scheduler"])
                self.init_step = train_state["step"]
                self.best_metric = train_state["best_metric"]
        elif self.config.model.get("_from_pretrained_") and self.config.tokenizer.get("_from_pretrained_"):
            self.from_pretrained()
            self.config.set_vocab_size(self.tokenizer.vocab_size)
            self.init_data()
        else:
            self.tokenizer = get_tokenizer(
                self.config.tokenizer.get("_tokenizer_name_"))
            self.init_data()
            self.model = instantiate(self.config.model)
            self.model.to(self.device)
            if self.config.base.get("train"):
                self.optimizer = instantiate(
                    self.config["optimizer"])(self.model.parameters())
                self.lr_scheduler = instantiate(self.config["scheduler"])(
                    optimizer=self.optimizer,
                    num_training_steps=self.total_step
                )


    def init_data(self):
        self.data_factory = DataFactory(tokenizer=self.tokenizer,
                                        use_multi_intent=self.config.base.get("multi_intent"),
                                        to_lower_case=self.config.tokenizer.get("_to_lower_case_"))
        batch_size = self.config.base["batch_size"]
        # init tokenizer config and dataloaders
        tokenizer_config = {key: self.config.tokenizer[key]
                            for key in self.config.tokenizer if key[0] != "_" and key[-1] != "_"}
        
        if self.config.base.get("train"):
            # init dataloader & load data
            
            
            train_dataset = self.data_factory.load_dataset(self.config.dataset, split="train")

            # update label and vocabulary (ONLY SUPPORT FOR "word_tokenizer")
            self.data_factory.update_label_names(train_dataset)
            self.data_factory.update_vocabulary(train_dataset)

            
            self.train_dataloader = self.data_factory.get_data_loader(train_dataset,
                                                       batch_size,
                                                       shuffle=True,
                                                       device=self.device,
                                                       enable_label=True,
                                                       align_mode=self.config.tokenizer.get(
                                                           "_align_mode_"),
                                                       label2tensor=True,
                                                       **tokenizer_config)
            self.total_step = int(self.config.base.get("epoch_num")) * len(self.train_dataloader)
            if self.saver.use_validation():
                dev_dataset = self.data_factory.load_dataset(self.config.dataset, split="validation")
                self.dev_dataloader = self.data_factory.get_data_loader(dev_dataset,
                                                        batch_size,
                                                        shuffle=False,
                                                        device=self.device,
                                                        enable_label=True,
                                                        align_mode=self.config.tokenizer.get(
                                                            "_align_mode_"),
                                                        label2tensor=False,
                                                        **tokenizer_config)
                self.data_factory.update_vocabulary(dev_dataset)
            self.intent_list = None
            self.intent_dict = None
            self.slot_list = None
            self.slot_dict = None
            # add intent label num and slot label num to config
            if self.config.model["decoder"].get("intent_classifier") and int(self.config.get_intent_label_num()) == 0:
                self.intent_list = self.data_factory.intent_label_list
                self.intent_dict = self.data_factory.intent_label_dict
                self.config.set_intent_label_num(len(self.intent_list))
            if self.config.model["decoder"].get("slot_classifier") and int(self.config.get_slot_label_num()) == 0:
                self.slot_list = self.data_factory.slot_label_list
                self.slot_dict = self.data_factory.slot_label_dict
                self.config.set_slot_label_num(len(self.slot_list))
                
            

            # autoload embedding for non-pretrained encoder
            if self.config["model"]["encoder"].get("embedding") and self.config["model"]["encoder"]["embedding"].get(
                    "load_embedding_name"):
                self.config["model"]["encoder"]["embedding"]["embedding_matrix"] = load_embedding(self.tokenizer,
                                                                                                  self.config["model"][
                                                                                                      "encoder"][
                                                                                                      "embedding"].get(
                                                                                                      "load_embedding_name"))
            # fill template in config
            self.config.autoload_template()
            # save config
            self.logger.set_config(self.config)
            self.saver.save_tokenizer(self.tokenizer)
            self.saver.save_label(self.intent_list, self.slot_list)
            self.config.set_vocab_size(self.tokenizer.vocab_size)
            
        if self.config.base.get("test"):
            self.test_dataset = self.data_factory.load_dataset(self.config.dataset, split="test")
            self.test_dataloader = self.data_factory.get_data_loader(self.test_dataset,
                                                      batch_size,
                                                      shuffle=False,
                                                      device=self.device,
                                                      enable_label=True,
                                                      align_mode=self.config.tokenizer.get(
                                                          "_align_mode_"),
                                                      label2tensor=False,
                                                      **tokenizer_config)

    def eval(self, step: int, best_metric: float) -> float:
        """ evaluation models.

        Args:
            step (int): which step the model has trained in
            best_metric (float): last best metric value to judge whether to test or save model

        Returns:
            float: updated best metric value
        """
        # TODO: save dev
        _, res = self.__evaluate(self.model, self.dev_dataloader, mode="dev")
        self.logger.log_metric(res, metric_split="dev", step=step)
        if res[self.config.evaluator.get("best_key")] > best_metric:
            best_metric = res[self.config.evaluator.get("best_key")]
            train_state = {
                "step": step,
                "best_metric": best_metric,
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict()
            }
            self.saver.save_model(self.model, train_state, self.accelerator)
            if self.config.base.get("test"):
                outputs, test_res = self.__evaluate(self.model, self.test_dataloader, mode="test")
                self.saver.save_output(outputs, self.test_dataset)
                self.logger.log_metric(test_res, metric_split="test", step=step)
        return best_metric

    def train(self) -> float:
        """ train models.

        Returns:
            float: updated best metric value
        """
        self.model.train()
        if self.accelerator is not None:
            self.total_step = math.ceil(self.total_step / self.accelerator.num_processes)
        if self.optimizer is None:
            self.optimizer = instantiate(self.config["optimizer"])(self.model.parameters())
        if self.lr_scheduler is None:
            self.lr_scheduler = instantiate(self.config["scheduler"])(
                optimizer=self.optimizer,
                num_training_steps=self.total_step
            )
        if not self.prepared and self.accelerator is not None:
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.lr_scheduler)
        step = self.init_step
        progress_bar = tqdm(range(self.total_step))
        progress_bar.update(self.init_step)
        self.optimizer.zero_grad()
        for _ in range(int(self.config.base.get("epoch_num"))):
            
            for data in self.train_dataloader:
                if step == 0:
                    self.logger.info(data.get_item(
                        0, tokenizer=self.tokenizer, intent_map=self.intent_list, slot_map=self.slot_list))
                output = self.model(data)
                if self.accelerator is not None and hasattr(self.model, "module"):
                    loss, intent_loss, slot_loss = self.model.module.compute_loss(
                        pred=output, target=data)
                else:
                    loss, intent_loss, slot_loss = self.model.compute_loss(
                        pred=output, target=data)
                self.logger.log_loss(loss, "Loss", step=step)
                self.logger.log_loss(intent_loss, "Intent Loss", step=step)
                self.logger.log_loss(slot_loss, "Slot Loss", step=step)
                self.optimizer.zero_grad()

                if self.accelerator is not None:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                train_state = {
                    "step": step,
                    "best_metric": self.best_metric,
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict()
                }
                self.saver.auto_save_step(self.model, train_state, self.accelerator)
                if self.saver.use_validation():
                    if not self.config.evaluator.get("eval_by_epoch") and step % self.config.evaluator.get("eval_step") == 0 and step != 0:
                        self.best_metric = self.eval(step, self.best_metric)
                step += 1
                progress_bar.update(1)
            if self.saver.use_validation() and self.config.evaluator.get("eval_by_epoch"):
                self.best_metric = self.eval(step, self.best_metric)
        self.logger.finish()
        return self.best_metric

    def test(self):
        return self.__evaluate(self.model, self.test_dataloader, mode="test")
    
    def __set_seed(self, seed_value: int):
        """Manually set random seeds.

        Args:
            seed_value (int): random seed
        """
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.random.manual_seed(seed_value)
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
        return

    def __evaluate(self, model, dataloader, mode="dev"):
        with torch.no_grad():
            model.eval()
            inps = InputData()
            outputs = OutputData()
            for data in dataloader:
                torch.cuda.empty_cache()
                if self.accelerator is not None and hasattr(self.model, "module"):
                    output = model.module(data)
                    decode_output = model.module.decode(output, data)
                else:
                    output = model(data)
                    decode_output = model.decode(output, data)

                decode_output.map_output(slot_map=self.slot_list,
                                        intent_map=self.intent_list)
                if self.config.model["decoder"].get("slot_classifier"):
                    data, decode_output = utils.remove_slot_ignore_index(
                        data, decode_output, ignore_index="#")

                inps.merge_input_data(data)
                outputs.merge_output_data(decode_output)
            if "metric" in self.config.evaluator:
                res = Evaluator.compute_all_metric(
                    inps, outputs, intent_label_map=self.intent_dict, metric_list=self.config.evaluator["metric"])
            else:
                res = Evaluator.compute_all_metric(
                    inps, outputs, intent_label_map=self.intent_dict)
            self.logger.info(f"Best {mode} metric: "+str(res))
        model.train()
        return outputs, res

    def load(self):
        
        if self.tokenizer is None:
            with open(os.path.join(self.load_dir, "tokenizer.pkl"), 'rb') as f:
                self.tokenizer = dill.load(f)
        label = utils.load_json(os.path.join(self.load_dir, "label.json"))
        if label["intent"] is None:
            self.intent_list = None
            self.intent_dict = None
        else:
            self.intent_list = label["intent"]
            self.intent_dict = {x: i for i, x in enumerate(label["intent"])}
            self.config.set_intent_label_num(len(self.intent_list))
        if label["slot"] is None:
            self.slot_list = None
            self.slot_dict = None
        else:
            self.slot_list = label["slot"]
            self.slot_dict = {x: i for i, x in enumerate(label["slot"])}
            self.config.set_slot_label_num(len(self.slot_list))
        self.config.set_vocab_size(self.tokenizer.vocab_size)
        if self.accelerator is not None and self.load_dir is not None:
            self.model = torch.load(os.path.join(self.load_dir, "model.pkl"), map_location=torch.device(self.device))
            self.prepared = True
            self.accelerator.load_state(self.load_dir)
            self.accelerator.prepare_model(self.model)
        else:
            self.model = torch.load(os.path.join(
            self.load_dir, "model.pkl"), map_location=torch.device(self.device))
                # if self.config.tokenizer["_tokenizer_name_"] == "word_tokenizer":
                #     self.tokenizer = get_tokenizer_class(self.config.tokenizer["_tokenizer_name_"]).from_file(os.path.join(self.load_dir, "tokenizer.json"))
                # else:
                #     self.tokenizer = get_tokenizer(self.config.tokenizer["_tokenizer_name_"])
            self.model.to(self.device)
            

    def from_pretrained(self):
        self.config.autoload_template()
        model = PretrainedModelForSLU.from_pretrained(self.config.model["_from_pretrained_"])
        # model = load_model(self.config.model["_from_pretrained_"])
        self.model = model.model
        if self.tokenizer is None:
            self.tokenizer = PreTrainedTokenizerForSLU.from_pretrained(
                self.config.tokenizer["_from_pretrained_"])
            self.config.tokenizer = model.config.tokenizer
            # self.tokenizer = load_tokenizer(self.config.tokenizer["_from_pretrained_"])

        self.model.to(self.device)
        label = model.config._id2label
        self.config.model = model.config.model
        self.intent_list = label["intent"]
        self.slot_list = label["slot"]
        self.intent_dict = {x: i for i, x in enumerate(label["intent"])}
        self.slot_dict = {x: i for i, x in enumerate(label["slot"])}

    def predict(self, text_data):
        self.model.eval()
        tokenizer_config = {key: self.config.tokenizer[key]
                            for key in self.config.tokenizer if key[0] != "_" and key[-1] != "_"}
        align_mode = self.config.tokenizer.get("_align_mode_")
        inputs = self.data_factory.batch_fn(batch=[{"text": text_data.split(" ")}],
                                            device=self.device,
                                            config=tokenizer_config,
                                            enable_label=False,
                                            align_mode=align_mode if align_mode is not None else "general",
                                            label2tensor=False)
        output = self.model(inputs)
        decode_output = self.model.decode(output, inputs)
        decode_output.map_output(slot_map=self.slot_list,
                                 intent_map=self.intent_list)
        if self.config.base.get("multi_intent"):
            intent = decode_output.intent_ids[0]
        else:
            intent = [decode_output.intent_ids[0]]
        input_ids = inputs.input_ids[0].tolist()
        tokens = [self.tokenizer.decode(ids) for ids in input_ids]
        slots = decode_output.slot_ids[0]
        return {"intent": intent, "slot": slots, "text": tokens}
