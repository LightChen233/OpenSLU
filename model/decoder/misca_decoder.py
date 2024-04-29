'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-12-26 17:14:44
LastEditTime: 2023-12-28 13:37:56
Description: 

'''
from common.utils import HiddenData, OutputData
from model.decoder import BaseDecoder
class MISCADecoder(BaseDecoder):
    def forward(self, hidden: HiddenData):
        if self.interaction is not None:
            hidden = self.interaction(hidden)
        return OutputData(hidden.get_intent_hidden_state(), hidden.get_slot_hidden_state())
