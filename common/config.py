'''
Author: Qiguang Chen
Date: 2023-01-11 10:39:26
LastEditors: Qiguang Chen
LastEditTime: 2023-03-22 14:35:03
Description: Configuration class to manage all process in OpenSLU like model construction, learning processing and so on.

'''
import re

from ruamel import yaml
import datetime

class Config(dict):
    def __init__(self, *args, **kwargs):
        """ init with dict as args
        """
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
        self.start_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        if not self.model.get("_from_pretrained_"):
            self.__autowired()

    @staticmethod
    def load_from_yaml(file_path:str)->"Config":
        """load config files with path

        Args:
            file_path (str): yaml configuration file path.

        Returns:
            Config: config object.
        """
        with open(file_path) as stream:
            try:
                return Config(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)

    @staticmethod
    def load_from_args(args)->"Config":
        """ load args to replace item value in config files assigned with '--config_path' or '--model'

        Args:
            args (Any): args with command line.

        Returns:
            Config: _description_
        """
        if args.model is not None and args.dataset is not None:
            args.config_path = f"config/reproduction/{args.dataset}/{args.model}.yaml"
        config = Config.load_from_yaml(args.config_path)
        if args.dataset is not None:
            config.__update_dataset(args.dataset)
        if args.device is not None:
            config["base"]["device"] = args.device
        if args.learning_rate is not None:
            config["optimizer"]["lr"] = args.learning_rate
        if args.epoch_num is not None:
            config["base"]["epoch_num"] = args.epoch_num
        return config

    def autoload_template(self):
        """ search '{*}' template to excute as python code, support replace variable as any configure item
        """
        self.__autoload_template(self.__dict__)

    def __get_autoload_value(self, matched):
        keys = matched.group()[1:-1].split(".")
        temp = self.__dict__
        for k in keys:
            temp = temp[k]
        return str(temp)

    def __autoload_template(self, config:dict):
        for k in config:
            if isinstance(config, dict):
                sub_config = config[k]
            elif isinstance(config, list):
                sub_config = k
            else:
                continue
            if isinstance(sub_config, dict) or isinstance(sub_config, list):
                self.__autoload_template(sub_config)
            if isinstance(sub_config, str) and "{" in sub_config and "}" in sub_config:
                res = re.sub(r'{.*?}', self.__get_autoload_value, config[k])
                res_dict= {"res": None}
                exec("res=" + res, res_dict)
                config[k] = res_dict["res"]

    def __update_dataset(self, dataset_name):
        if dataset_name is not None and isinstance(dataset_name, str):
            self.__dict__["dataset"]["dataset_name"] = dataset_name

    def get_model_config(self):
        return self.__dict__["model"]

    def __autowired(self):
        # Set encoder
        encoder_config = self.__dict__["model"]["encoder"]
        encoder_type = encoder_config["_model_target_"].split(".")[-1]

        def get_output_dim(encoder_config):
            encoder_type = encoder_config["_model_target_"].split(".")[-1]
            if (encoder_type == "AutoEncoder" and encoder_config["encoder_name"] in ["lstm", "self-attention-lstm",
                                                                                     "bi-encoder"]) or encoder_type == "NoPretrainedEncoder":
                output_dim = 0
                if encoder_config.get("lstm"):
                    output_dim += encoder_config["lstm"]["output_dim"]
                if encoder_config.get("attention"):
                    output_dim += encoder_config["attention"]["output_dim"]
                return output_dim
            else:
                return encoder_config["output_dim"]

        if encoder_type == "BiEncoder":
            output_dim = get_output_dim(encoder_config["intent_encoder"]) + \
                         get_output_dim(encoder_config["slot_encoder"])
        else:
            output_dim = get_output_dim(encoder_config)
        self.__dict__["model"]["encoder"]["output_dim"] = output_dim

        # Set interaction
        if "interaction" in self.__dict__["model"]["decoder"] and self.__dict__["model"]["decoder"]["interaction"].get(
                "input_dim") is None:
            self.__dict__["model"]["decoder"]["interaction"]["input_dim"] = output_dim
            interaction_type = self.__dict__["model"]["decoder"]["interaction"]["_model_target_"].split(".")[-1]
            if not ((encoder_type == "AutoEncoder" and encoder_config[
                "encoder_name"] == "self-attention-lstm") or encoder_type == "SelfAttentionLSTMEncoder") and interaction_type != "BiModelWithoutDecoderInteraction":
                output_dim = self.__dict__["model"]["decoder"]["interaction"]["output_dim"]

        # Set classifier
        if "slot_classifier" in self.__dict__["model"]["decoder"]:
            if self.__dict__["model"]["decoder"]["slot_classifier"].get("input_dim") is None:
                self.__dict__["model"]["decoder"]["slot_classifier"]["input_dim"] = output_dim
            self.__dict__["model"]["decoder"]["slot_classifier"]["use_slot"] = True
        if "intent_classifier" in self.__dict__["model"]["decoder"]:
            if self.__dict__["model"]["decoder"]["intent_classifier"].get("input_dim") is None:
                self.__dict__["model"]["decoder"]["intent_classifier"]["input_dim"] = output_dim
            self.__dict__["model"]["decoder"]["intent_classifier"]["use_intent"] = True

    def get_intent_label_num(self):
        """ get the number of intent labels.
        """
        classifier_conf = self.__dict__["model"]["decoder"]["intent_classifier"]
        return classifier_conf["intent_label_num"] if "intent_label_num" in classifier_conf else 0

    def get_slot_label_num(self):
        """ get the number of slot labels.
        """
        classifier_conf = self.__dict__["model"]["decoder"]["slot_classifier"]
        return classifier_conf["slot_label_num"] if "slot_label_num" in classifier_conf else 0

    def set_intent_label_num(self, intent_label_num):
        """ set the number of intent labels.
        
        Args:
            slot_label_num (int): the number of intent label
        """
        self.__dict__["base"]["intent_label_num"] = intent_label_num
        self.__dict__["model"]["decoder"]["intent_classifier"]["intent_label_num"] = intent_label_num
        if "interaction" in self.__dict__["model"]["decoder"]:

            self.__dict__["model"]["decoder"]["interaction"]["intent_label_num"] = intent_label_num
            # if self.__dict__["model"]["decoder"]["interaction"]["_model_target_"].split(".")[
            #     -1] == "StackInteraction":
            #     self.__dict__["model"]["decoder"]["slot_classifier"]["input_dim"] += intent_label_num

    
    def set_slot_label_num(self, slot_label_num:int)->None:
        """set the number of slot label
        
        Args:
            slot_label_num (int): the number of slot label
        """
        self.__dict__["base"]["slot_label_num"] = slot_label_num
        self.__dict__["model"]["decoder"]["slot_classifier"]["slot_label_num"] = slot_label_num
        if "interaction" in self.__dict__["model"]["decoder"]:
            self.__dict__["model"]["decoder"]["interaction"]["slot_label_num"] = slot_label_num

    def set_vocab_size(self, vocab_size):
        """set the size of vocabulary in non-pretrained tokenizer
        Args:
            slot_label_num (int): the number of slot label
        """
        encoder_type = self.__dict__["model"]["encoder"]["_model_target_"].split(".")[-1]
        encoder_name = self.__dict__["model"]["encoder"].get("encoder_name")
        if encoder_type == "BiEncoder" or (encoder_type == "AutoEncoder" and encoder_name == "bi-encoder"):
            self.__dict__["model"]["encoder"]["intent_encoder"]["embedding"]["vocab_size"] = vocab_size
            self.__dict__["model"]["encoder"]["slot_encoder"]["embedding"]["vocab_size"] = vocab_size
        elif self.__dict__["model"]["encoder"].get("embedding"):
            self.__dict__["model"]["encoder"]["embedding"]["vocab_size"] = vocab_size
