import torch
import torch.nn.functional as F
from torch import nn

from common.utils import HiddenData
from model.decoder.interaction.base_interaction import BaseInteraction


class BiModelInteraction(BaseInteraction):
    def __init__(self, **config):
        super().__init__(**config)
        self.intent_lstm = nn.LSTM(input_size=self.config["input_dim"], hidden_size=self.config["output_dim"],
                                   batch_first=True,
                                   num_layers=1)
        self.slot_lstm = nn.LSTM(input_size=self.config["input_dim"] + self.config["output_dim"],
                                 hidden_size=self.config["output_dim"], num_layers=1)

    def forward(self, encode_hidden: HiddenData, **kwargs):
        slot_hidden = encode_hidden.get_slot_hidden_state()
        intent_hidden_detached = encode_hidden.get_intent_hidden_state().clone().detach()
        seq_lens = encode_hidden.inputs.attention_mask.sum(-1)
        batch = slot_hidden.size(0)
        length = slot_hidden.size(1)
        dec_init_out = torch.zeros(batch, 1, self.config["output_dim"]).to(slot_hidden.device)
        hidden_state = (torch.zeros(1, 1, self.config["output_dim"]).to(slot_hidden.device), torch.zeros(1, 1, self.config["output_dim"]).to(slot_hidden.device))
        slot_hidden = torch.cat((slot_hidden, intent_hidden_detached), dim=-1).transpose(1,
                                                                                         0)  # 50 x batch x feature_size
        slot_drop = F.dropout(slot_hidden, self.config["dropout_rate"])
        all_out = []
        for i in range(length):
            if i == 0:
                out, hidden_state = self.slot_lstm(torch.cat((slot_drop[i].unsqueeze(1), dec_init_out), dim=-1),
                                                   hidden_state)
            else:
                out, hidden_state = self.slot_lstm(torch.cat((slot_drop[i].unsqueeze(1), out), dim=-1), hidden_state)
            all_out.append(out)
        slot_output = torch.cat(all_out, dim=1)  # batch x 50 x feature_size

        intent_hidden = torch.cat((encode_hidden.get_intent_hidden_state(),
                                   encode_hidden.get_slot_hidden_state().clone().detach()),
                                  dim=-1)
        intent_drop = F.dropout(intent_hidden, self.config["dropout_rate"])
        intent_lstm_output, _ = self.intent_lstm(intent_drop)
        intent_output = F.dropout(intent_lstm_output, self.config["dropout_rate"])
        output_list = []
        for index, slen in enumerate(seq_lens):
            output_list.append(intent_output[index, slen - 1, :].unsqueeze(0))

        encode_hidden.update_intent_hidden_state(torch.cat(output_list, dim=0))
        encode_hidden.update_slot_hidden_state(slot_output)

        return encode_hidden


class BiModelWithoutDecoderInteraction(BaseInteraction):
    def forward(self, encode_hidden: HiddenData, **kwargs):
        slot_hidden = encode_hidden.get_slot_hidden_state()
        intent_hidden_detached = encode_hidden.get_intent_hidden_state().clone().detach()
        seq_lens = encode_hidden.inputs.attention_mask.sum(-1)
        slot_hidden = torch.cat((slot_hidden, intent_hidden_detached), dim=-1)  # 50 x batch x feature_size
        slot_output = F.dropout(slot_hidden, self.config["dropout_rate"])

        intent_hidden = torch.cat((encode_hidden.get_intent_hidden_state(),
                                   encode_hidden.get_slot_hidden_state().clone().detach()),
                                  dim=-1)
        intent_output = F.dropout(intent_hidden, self.config["dropout_rate"])
        output_list = []
        for index, slen in enumerate(seq_lens):
            output_list.append(intent_output[index, slen - 1, :].unsqueeze(0))

        encode_hidden.update_intent_hidden_state(torch.cat(output_list, dim=0))
        encode_hidden.update_slot_hidden_state(slot_output)

        return encode_hidden
