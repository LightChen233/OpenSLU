'''
Author: Qiguang Chen
Date: 2023-01-11 10:39:26
LastEditors: Qiguang Chen
LastEditTime: 2023-12-27 22:23:32
Description: 

'''
from torch import nn
import os
from typing import Callable, Optional
import torch
from common.module_utils.common_utils import LSTMEncoder
from common.utils import HiddenData
import torch.nn.functional as F
from common.utils import HiddenData, OutputData, InputData

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
            torch.nn.init.normal(first_linear.weight, mean, std)
            if first_linear.bias is not None:
                first_linear.bias.data.fill_(0)

        for linear in self.second_linears:
            torch.nn.init.normal(linear.weight, mean, std)
            if linear.bias is not None:
                linear.bias.data.fill_(0)
        for linear in self.third_linears:
            torch.nn.init.normal(linear.weight, mean, std)

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

        return context_vector, weighted_output, att_weights

    # Using when use_regularisation = True
    @staticmethod
    def l2_matrix_norm(m):
        """
        Frobenius norm calculation
        :param m: {Variable} ||AAT - I||
        :return: regularized value
        """
        return torch.sum(torch.sum(torch.sum(m ** 2, 1), 1) ** 0.5)

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
        attention_weights = None
        previous_level_projection = None
        weighted_outputs = []
        attention_weights = []
        context_vectors = []
        for level in range(n_levels):
            context_vector, weighted_output, attention_weight = self.attn(all_output, previous_level_projection, label_level=level)

            previous_level_projection = self.projection_linears[level](torch.sigmoid(weighted_output))
            previous_level_projection = F.sigmoid(previous_level_projection)
            weighted_outputs.append(weighted_output)
            attention_weights.append(attention_weight)
            context_vectors.append(context_vector)
            
        return context_vectors, weighted_outputs, attention_weights

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
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.linear_slot(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x, self.linear(x)
        # return x

def init_attention_layer(model, name, n_labels, n_levels, output_size):
    
    model.level_projection_size = model.args.level_projection_size
    if model.attention_mode is not None:
        model.add_module(f'attention_{name}', AttentionLayer(args=model.args, size=output_size,
                                                            level_projection_size=model.level_projection_size,
                                                            n_labels=n_labels, n_level=n_levels))
    linears = []
    projection_linears = []
    for level in range(n_levels):
        level_projection_size = 0 if level == 0 else model.level_projection_size
        linears.append(nn.Linear(output_size + level_projection_size,
                                    n_labels[level]))
        projection_linears.append(nn.Linear(n_labels[level], model.level_projection_size, bias=False))
    model.add_module(f'linears_{name}', nn.ModuleList(linears))
    model.add_module(f'projection_linears_{name}', nn.ModuleList(projection_linears))
   


def perform_attention(model, name, all_output, last_output, n_labels, n_levels):
    attention_weights = None
    previous_level_projection = None
    weighted_outputs = []
    attention_weights = []
    context_vectors = []
    for level in range(n_levels):
        context_vector, weighted_output, attention_weight = model.__getattr__(f'attention_{name}')(all_output,
                                                            previous_level_projection, label_level=level)

        previous_level_projection = model.__getattr__(f'projection_linears_{name}')[level](
            torch.sigmoid(weighted_output) if model.attention_mode in ["label", "caml"]
            else torch.softmax(weighted_output, 1))
        previous_level_projection = F.sigmoid(previous_level_projection)
        weighted_outputs.append(weighted_output)
        attention_weights.append(attention_weight)
        context_vectors.append(context_vector)
        
    return context_vectors, weighted_outputs, attention_weights

class MISCADecoderV1(nn.Module):
    """Base class for all decoder module. 
    
    Notice: t is often only necessary to change this module and its sub-modules
    """
    def __init__(self, intent_classifier=None, slot_classifier=None, interaction=None, **config):
        super().__init__()
        self.config = config
        self.attn_type = args.intent_slot_attn_type
        self.n_levels = args.n_levels
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.slot_hier = [len(x) for x in slot_hier]

        self.lstm_intent = LSTMEncoder(
            config.hidden_size,
            args.decoder_hidden_dim,
            args.dropout_rate
        )
        self.lstm_slot = LSTMEncoder(
            config.hidden_size,
            args.decoder_hidden_dim,
            args.dropout_rate
        )

        self.intent_detection = IntentClassifier(self.num_intent_labels, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(
            args.decoder_hidden_dim,
            self.num_intent_labels,
            self.num_slot_labels,
            self.args.max_seq_len,
            self.args.slot_decoder_size,
            args.dropout_rate,
        )
        self.output_size = args.decoder_hidden_dim
        self.attention_mode = args.attention_mode
        
        if args.intent_slot_attn_type == 'coattention':
            dims = [self.args.label_embedding_size] + [args.slot_decoder_size] + [args.slot_decoder_size + args.level_projection_size] * (len(self.slot_hier) - 1) + [self.args.label_embedding_size]
            self.attn = HierCoAttention(args, dims, args.intent_slot_attn_size, args.dropout_rate)
        if args.intent_slot_attn_type:
            self.intent_refine = nn.Linear(args.decoder_hidden_dim + args.intent_slot_attn_size, self.num_intent_labels, args.dropout_rate)
            self.slot_refine = IntentClassifier(args.slot_decoder_size + args.intent_slot_attn_size, self.num_slot_labels, args.dropout_rate)
            self.slot_proj = IntentClassifier(self.num_slot_labels, self.args.label_embedding_size, args.dropout_rate)
            self.intent_proj = IntentClassifier(1, self.args.label_embedding_size, args.dropout_rate)

        init_attention_layer(self, 'intent', [self.num_intent_labels], 1, args.decoder_hidden_dim)
        if args.intent_slot_attn_type == 'coattention':
            init_attention_layer(self, 'slot', self.slot_hier, len(self.slot_hier), self.args.slot_decoder_size)

        self.relu = nn.LeakyReLU(0.2)
        self.intent_classifier = nn.Linear(args.decoder_hidden_dim, 1, args.dropout_rate)

    def sequence_mask(self, length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        mask = x.unsqueeze(0) < length.unsqueeze(1)
        # mask[:, 0] = 0
        return mask

    def forward(self, hidden: HiddenData):
        seq_lens = torch.sum(hidden.inputs.attention_mask, dim=-1).cpu()
        encoded = hidden.get_slot_hidden_state()
        intent_output = self.lstm_intent(encoded, seq_lens)
        slot_output = self.lstm_slot(encoded, seq_lens)

        intent_output = torch.cat(
            [torch.index_select(intent_output[i], 0, heads[i]).unsqueeze(0) for i in range(intent_output.size(0))],
            dim=0
        )
        slot_output = torch.cat(
            [torch.index_select(slot_output[i], 0, heads[i]).unsqueeze(0) for i in range(slot_output.size(0))],
            dim=0
        )

        i_context_vector, intent_logits, i_attn = perform_attention(self, 'intent', intent_output, None, [self.num_intent_labels], 1)
        intent_logits = intent_logits[-1]
        
        i_context_vector = i_context_vector[-1]
        intent_dec = self.intent_detection(intent_logits)
        x, slot_logits = self.slot_classifier(slot_output)

        if self.args.intent_slot_attn_type == 'coattention':
            s_context_vector, s_logits, s_attn = perform_attention(self, 'slot', x, None, self.slot_hier, len(self.slot_hier))

        if self.attn_type == 'coattention':
            if self.args.embedding_type == 'soft':
                slots = self.slot_proj(F.softmax(slot_logits, -1))
                intents = self.intent_proj(F.sigmoid(intent_logits.unsqueeze(2)))
            else:
                slot_label = torch.argmax(slot_logits, dim=-1)
                hard_label = F.one_hot(slot_label, num_classes=self.num_slot_labels)
                for i in range(len(seq_lens)):
                    hard_label[i, seq_lens[i]:, :] = 0
                slots = self.slot_proj(hard_label.float())
                
                int_labels = torch.zeros_like(intent_logits)
                num = torch.argmax(intent_dec, dim=-1)
                for i in range(len(intent_logits)):
                    num_i = num[i]
                    ids = torch.topk(intent_logits[i], num_i).indices
                    int_labels[i, ids] = 1.0
                
                intents = self.intent_proj(int_labels.unsqueeze(2))
            intent_vec, slot_vec = self.attn([intents] + s_context_vector + [slots])

        if self.attn_type:
            intent_logits = self.intent_refine.weight.mul(torch.tanh(torch.cat([i_context_vector, intent_vec], dim=-1))).sum(dim=2).add(self.intent_refine.bias)
            slot_logits = self.relu(self.slot_refine(torch.cat([x, self.relu(slot_vec)], dim=-1)))

        return OutputData(intent_logits, slot_logits)

    def decode(self, output: OutputData, target: InputData = None):
        """decode output logits

        Args:
            output (OutputData): output logits data
            target (InputData, optional): input data with attention mask. Defaults to None.

        Returns:
            List: decoded sequence ids
        """
        intent, slot = None, None
        if self.intent_classifier is not None:
            intent = self.intent_classifier.decode(output, target)
        if self.slot_classifier is not None:
            slot = self.slot_classifier.decode(output, target)
        return OutputData(intent, slot)

    def compute_loss(self, pred: OutputData, target: InputData, compute_intent_loss=True, compute_slot_loss=True):
        """compute loss.
        Notice: can set intent and slot loss weight by adding 'weight' config item in corresponding classifier configuration.

        Args:
            pred (OutputData): output logits data
            target (InputData): input golden data
            compute_intent_loss (bool, optional): whether to compute intent loss. Defaults to True.
            compute_slot_loss (bool, optional): whether to compute intent loss. Defaults to True.

        Returns:
            Tensor: loss result
        """
        seq_lens = torch.sum(target.attention_mask, dim=-1).cpu()
        max_len = torch.max(seq_lens)
        attention_mask = self.sequence_mask(seq_lens, max_length=max_len)
        total_loss = 0
        aux_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(pred.intent_ids.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.BCEWithLogitsLoss()
                intent_loss_cnt = nn.CrossEntropyLoss()
                intent_count = torch.sum(intent_label_ids, dim=-1).long()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.float()) 
                count_loss = intent_loss_cnt(intent_dec.view(-1, self.num_intent_labels), intent_count)
            total_loss += (intent_loss + count_loss) * self.args.intent_loss_coef 

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte().to(slot_logits.device), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    # print("SHAPE", slot_labels_ids.shape, slot_logits.shape, active_loss.shape)
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.reshape(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += slot_loss * (1 - self.args.intent_loss_coef)
        return total_loss, intent_loss, slot_loss