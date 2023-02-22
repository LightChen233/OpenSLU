'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-02-13 10:44:39
LastEditTime: 2023-02-14 10:28:43
Description: 

'''

import os
import dill
from common import utils
from common.utils import InputData, download
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer


# parser = argparse.ArgumentParser()
# parser.add_argument('--config_path', '-cp', type=str, default="config/reproduction/atis/joint_bert.yaml")
# args = parser.parse_args()
# config = Config.load_from_yaml(args.config_path)
# config.base["train"] = False
# config.base["test"] = False

# model_manager = ModelManager(config)
# model_manager.load()


class PretrainedConfigForSLU(PretrainedConfig):
    def __init__(self, **kargs) -> None:
        super().__init__(**kargs)

# pretrained_config = PretrainedConfigForSLU()
# # pretrained_config.push_to_hub("xxxx")


class PretrainedModelForSLU(PreTrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.config_class = config
        self.model = utils.instantiate(config.model)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        cls.config_class = PretrainedConfigForSLU
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class PreTrainedTokenizerForSLU(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        dir_names = f"save/{pretrained_model_name_or_path}".split("/")
        dir_name = ""
        for name in dir_names:
            dir_name +=  name+"/"
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
        cache_path = f"./save/{pretrained_model_name_or_path}/tokenizer.pkl"
        if not os.path.exists(cache_path):
            download(f"https://huggingface.co/{pretrained_model_name_or_path}/resolve/main/tokenizer.pkl", cache_path)
        with open(cache_path, "rb") as f:
            tokenizer = dill.load(f)
        return tokenizer


# pretrained_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# pretrained_tokenizer = PreTrainedTokenizerForSLU.from_pretrained("LightChen2333/joint-bert-slu-atis")
# test_model = PretrainedModelForSLU.from_pretrained("LightChen2333/joint-bert-slu-atis")
# print(test_model(InputData([pretrained_tokenizer("I want to go to Beijing !")])))