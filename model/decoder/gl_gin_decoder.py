import torch
import torch.nn.functional as F
from torch import nn

from common.utils import HiddenData, OutputData, InputData
from model.decoder import BaseDecoder
from model.decoder.interaction.gl_gin_interaction import LSTMEncoder


class IntentEncoder(nn.Module):
    def __init__(self,input_dim, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.__intent_lstm = LSTMEncoder(
            input_dim,
            input_dim,
            dropout_rate
        )

    def forward(self, g_hiddens, seq_lens):
        intent_lstm_out = self.__intent_lstm(g_hiddens, seq_lens)
        return F.dropout(intent_lstm_out, p=self.dropout_rate, training=self.training)


class GLGINDecoder(BaseDecoder):
    def __init__(self, intent_classifier, slot_classifier, interaction=None, **config):
        super().__init__(intent_classifier, slot_classifier, interaction)
        self.config=config
        self.intent_encoder = IntentEncoder(self.intent_classifier.config["input_dim"], self.config["dropout_rate"])

    def forward(self, hidden: HiddenData, forced_slot=None, forced_intent=None, differentiable=None):
        seq_lens = hidden.inputs.attention_mask.sum(-1)
        intent_lstm_out = self.intent_encoder(hidden.slot_hidden, seq_lens)
        hidden.update_intent_hidden_state(intent_lstm_out)
        pred_intent = self.intent_classifier(hidden)
        intent_index = self.intent_classifier.decode(OutputData(pred_intent, None),hidden.inputs,
                                                     return_list=False,
                                                     return_sentence_level=True)
        slot_hidden = self.interaction(
            hidden,
            pred_intent=pred_intent,
            intent_index=intent_index,
        )
        pred_slot = self.slot_classifier(slot_hidden)
        num_intent = self.intent_classifier.config["intent_label_num"]
        pred_slot = pred_slot.classifier_output[:, num_intent:]
        return OutputData(pred_intent, F.log_softmax(pred_slot, dim=1))