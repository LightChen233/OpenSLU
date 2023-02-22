import math

import einops
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from common.utils import HiddenData
from model.decoder.interaction import BaseInteraction


class SlotGatedInteraction(BaseInteraction):
    def __init__(self, **config):
        super().__init__(**config)
        self.intent_linear = nn.Linear(self.config["input_dim"],1, bias=False)
        self.slot_linear1 = nn.Linear(self.config["input_dim"],1, bias=False)
        self.slot_linear2 = nn.Linear(self.config["input_dim"],1, bias=False)
        self.remove_slot_attn = self.config["remove_slot_attn"]
        self.slot_gate = SlotGate(**config)

    def forward(self, encode_hidden: HiddenData, **kwargs):
        input_hidden = encode_hidden.get_slot_hidden_state()

        seq_lens = encode_hidden.inputs.attention_mask.sum(-1)
        output_list = []
        for index, slen in enumerate(seq_lens):
            output_list.append(input_hidden[index, slen - 1, :].unsqueeze(0))
        intent_input = torch.cat(output_list, dim=0)
        e_I = torch.tanh(self.intent_linear(intent_input)).squeeze(1)
        alpha_I = einops.repeat(e_I, 'b -> b h', h=intent_input.shape[-1])
        c_I = alpha_I * intent_input
        intent_hidden = intent_input+c_I
        if not self.remove_slot_attn:
            # slot attention
            h_k = einops.repeat(self.slot_linear1(input_hidden), 'b l h -> b l c h', c=input_hidden.shape[1])
            h_i = einops.repeat(self.slot_linear2(input_hidden), 'b l h -> b l c h', c=input_hidden.shape[1]).transpose(1,2)
            e_S = torch.tanh(h_k + h_i)
            alpha_S = torch.softmax(e_S, dim=2).squeeze(3)
            alpha_S = einops.repeat(alpha_S, 'b l1 l2 -> b l1 l2 h', h=input_hidden.shape[-1])
            map_input_hidden = einops.repeat(input_hidden, 'b l h -> b l c h', c=input_hidden.shape[1])
            c_S = torch.sum(alpha_S * map_input_hidden, dim=2)
        else:
            c_S = input_hidden
        slot_hidden = input_hidden + c_S * self.slot_gate(c_S,c_I)
        encode_hidden.update_intent_hidden_state(intent_hidden)
        encode_hidden.update_slot_hidden_state(slot_hidden)
        return encode_hidden

class SlotGate(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.linear = nn.Linear(config["input_dim"], config["output_dim"],bias=False)
        self.v = nn.Parameter(torch.rand(size=[1]))

    def forward(self, slot_context, intent_context):
        intent_gate = self.linear(intent_context)
        intent_gate = einops.repeat(intent_gate, 'b h -> b l h', l=slot_context.shape[1])
        return self.v * torch.tanh(slot_context + intent_gate)
