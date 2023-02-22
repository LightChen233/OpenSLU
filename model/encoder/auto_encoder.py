'''
Author: Qiguang Chen
Date: 2023-01-11 10:39:26
LastEditors: Qiguang Chen
LastEditTime: 2023-02-18 19:33:34
Description: 

'''
from common.utils import InputData
from model.encoder.base_encoder import BaseEncoder, BiEncoder
from model.encoder.pretrained_encoder import PretrainedEncoder
from model.encoder.non_pretrained_encoder import NonPretrainedEncoder

class AutoEncoder(BaseEncoder):
    
    def __init__(self, **config):
        """automatedly load encoder by 'encoder_name'
        Args:
            config (dict):
                encoder_name (str): support ["lstm", "self-attention-lstm", "bi-encoder"] and other pretrained model in hugging face
                **args (Any): other configuration items corresponding to each module.
        """
        super().__init__()
        self.config = config
        if config.get("encoder_name"):
            encoder_name = config.get("encoder_name").lower()
            if encoder_name in ["lstm", "self-attention-lstm"]:
                self.__encoder = NonPretrainedEncoder(**config)
            elif encoder_name == "bi-encoder":
                self.__encoder= BiEncoder(self.__init__(**config["intent_encoder"]), self.__init__(**config["intent_encoder"]))
            else:
                self.__encoder = PretrainedEncoder(**config)
        else:
            raise ValueError("There is no Encoder Name in config.")

    def forward(self, inputs: InputData):
        return self.__encoder(inputs)