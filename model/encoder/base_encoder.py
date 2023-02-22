'''
Author: Qiguang Chen
Date: 2023-01-11 10:39:26
LastEditors: Qiguang Chen
LastEditTime: 2023-01-26 17:25:17
Description: Base encoder and bi encoder

'''
from torch import nn

from common.utils import InputData


class BaseEncoder(nn.Module):
    """Base class for all encoder module
    """
    def __init__(self, **config):
        super().__init__()
        self.config = config
        NotImplementedError("no implement")

    def forward(self, inputs: InputData):
        self.encoder(inputs.input_ids)


class BiEncoder(nn.Module):
    """Bi Encoder for encode intent and slot separately
    """
    def __init__(self, intent_encoder: BaseEncoder, slot_encoder: BaseEncoder, **config):
        super().__init__()
        self.intent_encoder = intent_encoder
        self.slot_encoder = slot_encoder

    def forward(self, inputs: InputData):
        hidden_slot = self.slot_encoder(inputs)
        hidden_intent = self.intent_encoder(inputs)
        if not self.intent_encoder.config["return_sentence_level_hidden"]:
            hidden_slot.update_intent_hidden_state(hidden_intent.get_slot_hidden_state())
        else:
            hidden_slot.update_intent_hidden_state(hidden_intent.get_intent_hidden_state())
        return hidden_slot
