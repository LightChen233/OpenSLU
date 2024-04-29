'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-12-26 17:14:44
LastEditTime: 2023-12-28 23:39:43
Description: 

'''
import os
from typing import Callable, Optional
import torch
from common.module_utils.common_utils import LSTMEncoder
from common.utils import HiddenData
import torch.nn.functional as F
import torch.nn as nn

from model.decoder.interaction.base_interaction import BaseInteraction

def get_slot_labels(slot_dict):
    return [label.strip() for label in slot_dict.keys()]

def get_clean_labels(slot_dict):
    return [label.strip() for label in slot_dict.keys()]

def get_slots_all(task, slot_dict):
    slot_labels = get_slot_labels(slot_dict)
    hier = ()
    if "atis" in task:
        slot_parents = get_clean_labels(slot_dict)
        hier = (slot_parents, )
    slot_type = sorted(set([name[2:] for name in slot_labels if name[:2] == 'B-' or name[:2] == 'I-']))
    hier += (slot_type, )
    return slot_labels, hier

class Biaffine(nn.Module):
    r"""
    Biaffine layer for first-order scoring :cite:`dozat-etal-2017-biaffine`.
    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y)` of the vector pair :math:`(x, y)` is computed as :math:`x^T W y / d^s`,
    where `d` and `s` are vector dimension and scaling factor respectively.
    :math:`x` and :math:`y` can be concatenated with bias terms.
    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        n_proj (Optional[int]):
            If specified, applies MLP layers to reduce vector dimensions. Default: ``None``.
        dropout (Optional[float]):
            If specified, applies a :class:`SharedDropout` layer with the ratio on MLP outputs. Default: 0.
        scale (float):
            Factor to scale the scores. Default: 0.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``True``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``True``.
        decompose (bool):
            If ``True``, represents the weight as the product of 2 independent matrices. Default: ``False``.
        init (Callable):
            Callable initialization method. Default: `nn.init.zeros_`.
    """

    def __init__(
        self,
        n_x: int,
        n_y: int,
        n_out: int = 1,
        dropout: Optional[float] = 0,
        scale: int = 0,
        bias_x: bool = False,
        bias_y: bool = False,
        init: Callable = nn.init.zeros_
    ):
        super().__init__()

        self.n_x = n_x
        self.n_y = n_y
        self.n_out = n_out
        self.dropout = dropout
        self.scale = scale
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.init = init

        # self.n_model = n_in
        self.weight = nn.Parameter(torch.Tensor(n_out, self.n_x + bias_x, self.n_y + bias_y))

        self.reset_parameters()

    def reset_parameters(self):
        self.init(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.
        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        return s.squeeze(1) / self.n_x ** self.scale

class HierCoAttention(nn.Module):
    def __init__(self, dims, out_dim, dropout_rate=0.):
        super(HierCoAttention, self).__init__()

        self.n_layers = len(dims)
        self.linears = nn.ModuleList([nn.Linear(inp_dim, out_dim, bias=True) for inp_dim in dims])
        self.reverse = nn.ModuleList([nn.Linear(inp_dim, out_dim, bias=True) for inp_dim in dims])

        self.scorers = nn.ModuleList([Biaffine(dims[i], dims[i + 1], dropout=dropout_rate) for i in range(self.n_layers - 1)])
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, inps):
        # inps should be list of [intent, ..., slots]
        assert len(inps) == self.n_layers
        Cs = []
        for i in range(self.n_layers - 1):
            Cs.append(self.scorers[i](inps[i], inps[i + 1]))
        
        projs = []
        revers = []
        for i in range(self.n_layers):
            projs.append(self.linears[i](inps[i]))
            revers.append(self.reverse[i](inps[i]))
        
        slots = None
        for i in range(self.n_layers - 1):
            if slots is None:
                slots = torch.tanh(torch.bmm(Cs[0].transpose(1, 2), projs[0]) + projs[1])
            else:
                slots = torch.bmm(Cs[i].transpose(1, 2), slots) + projs[i + 1]
                if i < self.n_layers - 2:
                    slots = torch.tanh(slots)
        # slots = self.dropout(slots)
        
        intents = None
        for i in range(self.n_layers - 1, 0, -1):
            if intents is None:
                intents = torch.tanh(torch.bmm(Cs[-1], revers[-1]) + revers[-2])
            else:
                intents = torch.bmm(Cs[i - 1], intents) + revers[i - 1]
                if i > 1:
                    intents = torch.tanh(intents)
        return intents, slots

class AttentionLayer(nn.Module):

    def __init__(self,
                 size: int,
                 level_projection_size: int = 0,
                 n_labels=None,
                 n_level: int = 1
                 ):
        """
        The init function
        :param args: the input parameters from commandline
        :param size: the input size of the layer, it is normally the output size of other DNN models,
            such as CNN, RNN
        """
        super(AttentionLayer, self).__init__()

        self.size = size
        # For self-attention: d_a and r are the dimension of the dense layer and the number of attention-hops
        # d_a is the output size of the first linear layer
        self.d_a = size

        # r is the number of attention heads

        self.n_labels = n_labels
        self.n_level = n_level

        self.level_projection_size = level_projection_size

        self.linear = nn.Linear(self.size, self.size, bias=False)
        
        self.first_linears = nn.ModuleList([nn.Linear(self.size, self.d_a, bias=False) for _ in range(self.n_level)])
        self.second_linears = nn.ModuleList([nn.Linear(self.d_a, self.n_labels[label_lvl], bias=False) for label_lvl in range(self.n_level)])
        self.third_linears = nn.ModuleList([nn.Linear(self.size +
                                            (self.level_projection_size if label_lvl > 0 else 0),
                                            self.n_labels[label_lvl], bias=True) for label_lvl in range(self.n_level)])

        self._init_weights(mean=0.0, std=0.03)

    def _init_weights(self, mean=0.0, std=0.03) -> None:
        """
        Initialise the weights
        :param mean:
        :param std:
        :return: None
        """
        for first_linear in self.first_linears:
            torch.nn.init.normal_(first_linear.weight, mean, std)
            if first_linear.bias is not None:
                first_linear.bias.data.fill_(0)

        for linear in self.second_linears:
            torch.nn.init.normal_(linear.weight, mean, std)
            if linear.bias is not None:
                linear.bias.data.fill_(0)
        for linear in self.third_linears:
            torch.nn.init.normal_(linear.weight, mean, std)

    def forward(self, x, previous_level_projection=None, label_level=0, masks=None):
        """
        :param x: [batch_size x max_len x dim (i.e., self.size)]

        :param previous_level_projection: the embeddings for the previous level output
        :param label_level: the current label level
        :return:
            Weighted average output: [batch_size x dim (i.e., self.size)]
            Attention weights
        """
        weights = F.tanh(self.first_linears[label_level](x))

        att_weights = self.second_linears[label_level](weights)
        att_weights = F.softmax(att_weights, 1).transpose(1, 2)
        if len(att_weights.size()) != len(x.size()):
            att_weights = att_weights.squeeze()
        context_vector = att_weights @ x
        
        batch_size = context_vector.size(0)

        if previous_level_projection is not None:
            temp = [context_vector,
                    previous_level_projection.repeat(1, self.n_labels[label_level]).view(batch_size, self.n_labels[label_level], -1)]
            context_vector = torch.cat(temp, dim=2)

        weighted_output = self.third_linears[label_level].weight.mul(context_vector).sum(dim=2).add(
            self.third_linears[label_level].bias)

        return context_vector, weighted_output

class MISCAAttention(nn.Module):
    def __init__(self, config, n_labels, n_levels, output_size) -> None:
        super().__init__()
        level_projection_size = config["level_projection_size"]
        self.attn = AttentionLayer(size=output_size,
                                    level_projection_size=config["level_projection_size"],
                                    n_labels=n_labels, n_level=n_levels)
        linears = []
        projection_linears = []
        for level in range(n_levels):
            level_projection_size = 0 if level == 0 else config["level_projection_size"]
            linears.append(nn.Linear(output_size + level_projection_size,
                                        n_labels[level]))
            projection_linears.append(nn.Linear(n_labels[level], config["level_projection_size"], bias=False))
        self.linears = nn.ModuleList(linears)
        self.projection_linears = nn.ModuleList(projection_linears)
    
    def forward(self, all_output, n_levels):
        previous_level_projection = None
        context_vectors = []
        for level in range(n_levels):
            context_vector, weighted_output = self.attn(all_output, previous_level_projection, label_level=level)
            previous_level_projection = self.projection_linears[level](torch.sigmoid(weighted_output))
            previous_level_projection = F.sigmoid(previous_level_projection)
            context_vectors.append(context_vector)
        return context_vectors

class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class SlotClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_slot_labels,
        attention_embedding_size=200,
        dropout_rate=0.0,
    ):
        super(SlotClassifier, self).__init__()
        self.num_slot_labels = num_slot_labels
        self.attention_embedding_size = attention_embedding_size

        output_dim = self.attention_embedding_size  # base model
        self.linear_slot = nn.Linear(input_dim, self.attention_embedding_size, bias=False)

        # output
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(output_dim, num_slot_labels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.linear_slot(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x, self.linear(x)
        # return x

class MISCAInteraction(BaseInteraction):
    def __init__(self, **config):
        super().__init__(**config)
        self.config=config
        self.lstm_intent = LSTMEncoder(
            self.config["input_dim"],
            self.config["hidden_dim"],
            self.config["dropout_rate"],
        )
        self.lstm_slot = LSTMEncoder(
            self.config["input_dim"],
            self.config["hidden_dim"],
            self.config["dropout_rate"],
        )
        self.attention_mode = self.config["attention_mode"]
        self.intent_refine = nn.Linear(self.config["hidden_dim"] + self.config["intent_slot_attn_size"], config["intent_label_num"], self.config["dropout_rate"])
        self.slot_refine = nn.Linear(self.config["hidden_dim"] + self.config["intent_slot_attn_size"], config["slot_label_num"], self.config["dropout_rate"])
        self.slot_proj = IntentClassifier(config["slot_label_num"], config["label_embedding_size"], self.config["dropout_rate"])
        self.intent_proj = IntentClassifier(1, config["label_embedding_size"], self.config["dropout_rate"])
        self.slot_classifier = SlotClassifier(
            config["hidden_dim"],
            config["slot_label_num"],
            config["slot_hidden_dim"],
            config["dropout_rate"],
        )
        self.relu = nn.LeakyReLU(0.2)
        self.embedding_type = self.config["embedding_type"]
        slot_hier = get_slots_all(config["task"], config["slot_dict"])
        self.slot_hier = [len(x) for x in slot_hier]
        self.intent_detection = IntentClassifier(config["intent_label_num"], config["intent_label_num"], self.config["dropout_rate"])
        dims = [config["label_embedding_size"]] + [self.config["hidden_dim"]] + [self.config["hidden_dim"] + self.config["level_projection_size"]] * (len(self.slot_hier) - 1) + [config["label_embedding_size"]]
        self.attn = HierCoAttention(dims, self.config["intent_slot_attn_size"], self.config["dropout_rate"])
        self.intent_attn = MISCAAttention(self.config, [self.config["intent_label_num"]], 1, self.config["hidden_dim"])
        self.slot_attn = MISCAAttention(self.config, self.slot_hier, len(self.slot_hier), self.config["hidden_dim"])
        
    def forward(self, encode_hidden: HiddenData, **kwargs):
        seq_lens = encode_hidden.inputs.attention_mask.sum(-1)
        intent_output = self.lstm_intent(encode_hidden.get_slot_hidden_state(), seq_lens)
        slot_output = self.lstm_slot(encode_hidden.get_slot_hidden_state(), seq_lens)


        i_context_vector = self.intent_attn(intent_output, 1)
        s_context_vector = self.slot_attn(slot_output, len(self.slot_hier))
        intent_vec, slot_vec = self.attn(i_context_vector + s_context_vector + [slot_output])

        intent_logits = self.intent_refine.weight.mul(torch.tanh(torch.cat([i_context_vector[0], intent_vec], dim=-1))).sum(dim=2).add(self.intent_refine.bias)
        slot_logits = self.slot_refine(torch.cat([slot_output, slot_vec], dim=-1))
        # intent_logits = self.intent_refine.weight.mul(torch.tanh(intent_output)).sum(dim=2).add(self.intent_refine.bias)
        # slot_logits = self.slot_refine(slot_output)
        encode_hidden.update_intent_hidden_state(intent_logits)
        encode_hidden.update_slot_hidden_state(slot_logits)
        return encode_hidden