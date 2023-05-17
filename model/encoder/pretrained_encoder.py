'''
Author: Qiguang Chen
Date: 2023-01-11 10:39:26
LastEditors: Qiguang Chen
LastEditTime: 2023-05-02 15:03:18
Description: pretrained encoder model

'''
from transformers import AutoModel, AutoConfig
from common import utils

from common.utils import InputData, HiddenData
from model.encoder.base_encoder import BaseEncoder


class PretrainedEncoder(BaseEncoder):
    def __init__(self, **config):
        """ init pretrained encoder
        
        Args:
            config (dict):
                encoder_name (str): pretrained model name in hugging face.
        """
        super().__init__(**config)
        if self.config.get("_is_check_point_"):
            self.encoder = utils.instantiate(config["pretrained_model"], target="_pretrained_model_target_")
            # print(self.encoder)
        else:
            self.encoder = AutoModel.from_pretrained(config["encoder_name"])

    def forward(self, inputs: InputData):
        output = self.encoder(**inputs.get_inputs())
        hidden = HiddenData(None, output.last_hidden_state)
        if self.config.get("return_with_input"):
            hidden.add_input(inputs)
        if self.config.get("return_sentence_level_hidden"):
            padding_side = self.config.get("padding_side")
            if hasattr(output, "pooler_output"):
                hidden.update_intent_hidden_state(output.pooler_output)
            elif padding_side is not None and padding_side == "left":
                hidden.update_intent_hidden_state(output.last_hidden_state[:, -1])
            else:
                hidden.update_intent_hidden_state(output.last_hidden_state[:, 0])
        else:
            hidden.update_intent_hidden_state(output.last_hidden_state)
        return hidden
