'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-02-13 10:44:39
LastEditTime: 2023-02-19 15:45:08
Description: 

'''

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dill

from common.config import Config
from common.model_manager import ModelManager
from transformers import PretrainedConfig, PreTrainedModel, AutoModel, AutoTokenizer, PreTrainedTokenizer

class PretrainedConfigForSLUToSave(PretrainedConfig):
    def __init__(self, **kargs) -> None:
        cfg = model_manager.config
        kargs["name_or_path"] = cfg.base["name"]
        kargs["return_dict"] = False
        kargs["is_decoder"] = True
        kargs["_id2label"] = {"intent": model_manager.intent_list, "slot": model_manager.slot_list}
        kargs["_label2id"] = {"intent": model_manager.intent_dict, "slot": model_manager.slot_dict}
        kargs["_num_labels"] = {"intent": len(model_manager.intent_list), "slot": len(model_manager.slot_list)}
        kargs["tokenizer_class"] = cfg.base["name"]
        kargs["vocab_size"] = model_manager.tokenizer.vocab_size
        kargs["model"] = cfg.model
        kargs["model"]["decoder"]["intent_classifier"]["intent_label_num"] = len(model_manager.intent_list)
        kargs["model"]["decoder"]["slot_classifier"]["slot_label_num"] = len(model_manager.slot_list)
        kargs["tokenizer"] = cfg.tokenizer
        len(model_manager.slot_list)
        super().__init__(**kargs)

class PretrainedModelForSLUToSave(PreTrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.model = model_manager.model
        self.config_class = config


class PreTrainedTokenizerForSLUToSave(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = model_manager.tokenizer
    
    # @overload
    def save_vocabulary(self, save_directory: str, filename_prefix = None):
        if filename_prefix is not None:
            path = os.path.join(save_directory, filename_prefix+"-tokenizer.pkl")
        else:
            path = os.path.join(save_directory, "tokenizer.pkl")
        # tokenizer_name=model_manager.config.tokenizer.get("_tokenizer_name_")
        # if tokenizer_name == "word_tokenizer":
        #     self.tokenizer.save(path)
        # else:
        #     torch.save()
        with open(path,'wb') as f:
            dill.dump(self.tokenizer,f)
        return (path,)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-cp', type=str, required=True)
    parser.add_argument('--output_path', '-op', type=str, default="save/temp")
    args = parser.parse_args()
    config = Config.load_from_yaml(args.config_path)
    config.base["train"] = False
    config.base["test"] = False
    if config.model_manager["load_dir"] is None:
        config.model_manager["load_dir"] = config.model_manager["save_dir"]
    model_manager = ModelManager(config)
    model_manager.load()
    model_manager.config.autoload_template()
    
    pretrained_config = PretrainedConfigForSLUToSave()
    pretrained_model= PretrainedModelForSLUToSave(pretrained_config)
    pretrained_model.save_pretrained(args.output_path)

    pretrained_tokenizer = PreTrainedTokenizerForSLUToSave()
    pretrained_tokenizer.save_pretrained(args.output_path)
