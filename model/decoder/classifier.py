'''
Author: Qiguang Chen
Date: 2023-01-11 10:39:26
LastEditors: Qiguang Chen
LastEditTime: 2023-01-31 20:07:00
Description: 

'''
import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from model.decoder import decoder_utils

from torchcrf import CRF

from common.utils import HiddenData, OutputData, InputData, ClassifierOutputData, unpack_sequence, pack_sequence, \
    instantiate


class BaseClassifier(nn.Module):
    """Base class for all classifier module
    """
    def __init__(self, **config):
        super().__init__()
        self.config = config
        if config.get("loss_fn"):
            self.loss_fn = instantiate(config.get("loss_fn"))
        else:
            self.loss_fn = CrossEntropyLoss(ignore_index=self.config.get("ignore_index"))

    def forward(self, *args, **kwargs):
        raise NotImplementedError("No implemented classifier.")

    def decode(self, output: OutputData,
               target: InputData = None,
               return_list=True,
               return_sentence_level=None):
        """decode output logits

        Args:
            output (OutputData): output logits data
            target (InputData, optional): input data with attention mask. Defaults to None.
            return_list (bool, optional):  if True return list else return torch Tensor.. Defaults to True.
            return_sentence_level (_type_, optional): if True decode sentence level intent else decode token level intent. Defaults to None.

        Returns:
            List or Tensor: decoded sequence ids
        """
        if self.config.get("return_sentence_level") is not None and return_sentence_level is None:
            return_sentence_level = self.config.get("return_sentence_level")
        elif self.config.get("return_sentence_level") is None and return_sentence_level is None:
            return_sentence_level = False
        return decoder_utils.decode(output, target,
                                    return_list=return_list,
                                    return_sentence_level=return_sentence_level,
                                    pred_type=self.config.get("mode"),
                                    use_multi=self.config.get("use_multi"),
                                    multi_threshold=self.config.get("multi_threshold"))

    def compute_loss(self, pred: OutputData, target: InputData):
        """compute loss

        Args:
            pred (OutputData): output logits data
            target (InputData): input golden data

        Returns:
            Tensor: loss result
        """
        _CRF = None
        if self.config.get("use_crf"):
            _CRF = self.CRF
        return decoder_utils.compute_loss(pred, target, criterion_type=self.config["mode"],
                                          use_crf=_CRF is not None,
                                          ignore_index=self.config["ignore_index"],
                                          use_multi=self.config.get("use_multi"),
                                          loss_fn=self.loss_fn,
                                          CRF=_CRF)


class LinearClassifier(BaseClassifier):
    """
    Decoder structure based on Linear.
    """
    def __init__(self, **config):
        """Construction function for LinearClassifier
        
        Args:
            config (dict):
                input_dim (int): hidden state dim.
                use_slot (bool): whether to classify slot label.
                slot_label_num (int, optional): the number of slot label. Enabled if use_slot is True.
                use_intent (bool): whether to classify intent label.
                intent_label_num (int, optional): the number of intent label. Enabled if use_intent is True.
                use_crf (bool): whether to use crf for slot.
        """
        super().__init__(**config)
        self.config = config
        if config.get("use_slot"):
            self.slot_classifier = nn.Linear(config["input_dim"], config["slot_label_num"])
            if self.config.get("use_crf"):
                self.CRF = CRF(num_tags=config["slot_label_num"], batch_first=True)
        if config.get("use_intent"):
            self.intent_classifier = nn.Linear(config["input_dim"], config["intent_label_num"])

    def forward(self, hidden: HiddenData):
        if self.config.get("use_intent"):
            return ClassifierOutputData(self.intent_classifier(hidden.get_intent_hidden_state()))
        if self.config.get("use_slot"):
            return ClassifierOutputData(self.slot_classifier(hidden.get_slot_hidden_state()))



class AutoregressiveLSTMClassifier(BaseClassifier):
    """
    Decoder structure based on unidirectional LSTM.
    """

    def __init__(self, **config):
        """ Construction function for Decoder.

        Args:
            config (dict):
                input_dim (int): input dimension of Decoder. In fact, it's encoder hidden size.
                use_slot (bool): whether to classify slot label.
                slot_label_num (int, optional): the number of slot label. Enabled if use_slot is True.
                use_intent (bool): whether to classify intent label.
                intent_label_num (int, optional): the number of intent label. Enabled if use_intent is True.
                use_crf (bool): whether to use crf for slot.
                hidden_dim (int): hidden dimension of iterative LSTM.
                embedding_dim (int): if it's not None, the input and output are relevant.
                dropout_rate (float): dropout rate of network which is only useful for embedding.
        """

        super(AutoregressiveLSTMClassifier, self).__init__(**config)
        if config.get("use_slot") and config.get("use_crf"):
            self.CRF = CRF(num_tags=config["slot_label_num"], batch_first=True)
        self.input_dim = config["input_dim"]
        self.hidden_dim = config["hidden_dim"]
        if config.get("use_intent"):
            self.output_dim = config["intent_label_num"]
        if config.get("use_slot"):
            self.output_dim = config["slot_label_num"]
        self.dropout_rate = config["dropout_rate"]
        self.embedding_dim = config.get("embedding_dim")
        self.force_ratio = config.get("force_ratio")
        self.config = config
        self.ignore_index = config.get("ignore_index") if config.get("ignore_index") is not None else -100
        # If embedding_dim is not None, the output and input
        # of this structure is relevant.
        if self.embedding_dim is not None:
            self.embedding_layer = nn.Embedding(self.output_dim, self.embedding_dim)
            self.init_tensor = nn.Parameter(
                torch.randn(1, self.embedding_dim),
                requires_grad=True
            )

        # Make sure the input dimension of iterative LSTM.
        if self.embedding_dim is not None:
            lstm_input_dim = self.input_dim + self.embedding_dim
        else:
            lstm_input_dim = self.input_dim

        # Network parameter definition.
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.lstm_layer = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=self.config["bidirectional"],
            dropout=self.dropout_rate,
            num_layers=self.config["layer_num"]
        )
        self.linear_layer = nn.Linear(
            self.hidden_dim,
            self.output_dim
        )
        # self.loss_fn = CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, hidden: HiddenData, internal_interaction=None, **interaction_args):
        """ Forward process for decoder.

        :param internal_interaction:
        :param hidden:
        :return: is distribution of prediction labels.
        """
        input_tensor = hidden.slot_hidden
        seq_lens = hidden.inputs.attention_mask.sum(-1).detach().cpu().tolist()
        output_tensor_list, sent_start_pos = [], 0
        input_tensor = pack_sequence(input_tensor, seq_lens)
        forced_input = None
        if self.training:
            if random.random() < self.force_ratio:
                if self.config["mode"]=="slot":

                    forced_slot = pack_sequence(hidden.inputs.slot, seq_lens)
                    temp_slot = []
                    for index, x in enumerate(forced_slot):
                        if index == 0:
                            temp_slot.append(x.reshape(1))
                        elif x == self.ignore_index:
                            temp_slot.append(temp_slot[-1])
                        else:
                            temp_slot.append(x.reshape(1))
                    forced_input = torch.cat(temp_slot, 0)
                if self.config["mode"]=="token-level-intent":
                    forced_intent = hidden.inputs.intent.unsqueeze(1).repeat(1, hidden.inputs.slot.shape[1])
                    forced_input = pack_sequence(forced_intent, seq_lens)
        if self.embedding_dim is None or forced_input is not None:

            for sent_i in range(0, len(seq_lens)):
                sent_end_pos = sent_start_pos + seq_lens[sent_i]

                # Segment input hidden tensors.
                seg_hiddens = input_tensor[sent_start_pos: sent_end_pos, :]

                if self.embedding_dim is not None and forced_input is not None:
                    if seq_lens[sent_i] > 1:
                        seg_forced_input = forced_input[sent_start_pos: sent_end_pos]

                        seg_forced_tensor = self.embedding_layer(seg_forced_input)[:-1]
                        seg_prev_tensor = torch.cat([self.init_tensor, seg_forced_tensor], dim=0)
                    else:
                        seg_prev_tensor = self.init_tensor

                    # Concatenate forced target tensor.
                    combined_input = torch.cat([seg_hiddens, seg_prev_tensor], dim=1)
                else:
                    combined_input = seg_hiddens
                dropout_input = self.dropout_layer(combined_input)
                lstm_out, _ = self.lstm_layer(dropout_input.view(1, seq_lens[sent_i], -1))
                if internal_interaction is not None:
                    interaction_args["sent_id"] = sent_i
                    lstm_out = internal_interaction(torch.transpose(lstm_out, 0, 1), **interaction_args)[:, 0]
                linear_out = self.linear_layer(lstm_out.view(seq_lens[sent_i], -1))

                output_tensor_list.append(linear_out)
                sent_start_pos = sent_end_pos
        else:
            for sent_i in range(0, len(seq_lens)):
                prev_tensor = self.init_tensor

                # It's necessary to remember h and c state
                # when output prediction every single step.
                last_h, last_c = None, None

                sent_end_pos = sent_start_pos + seq_lens[sent_i]
                for word_i in range(sent_start_pos, sent_end_pos):
                    seg_input = input_tensor[[word_i], :]
                    combined_input = torch.cat([seg_input, prev_tensor], dim=1)
                    dropout_input = self.dropout_layer(combined_input).view(1, 1, -1)
                    if last_h is None and last_c is None:
                        lstm_out, (last_h, last_c) = self.lstm_layer(dropout_input)
                    else:
                        lstm_out, (last_h, last_c) = self.lstm_layer(dropout_input, (last_h, last_c))

                    if internal_interaction is not None:
                        interaction_args["sent_id"] = sent_i
                        lstm_out = internal_interaction(lstm_out, **interaction_args)[:, 0]

                    lstm_out = self.linear_layer(lstm_out.view(1, -1))
                    output_tensor_list.append(lstm_out)

                    _, index = lstm_out.topk(1, dim=1)
                    prev_tensor = self.embedding_layer(index).view(1, -1)
                sent_start_pos = sent_end_pos
        seq_unpacked = unpack_sequence(torch.cat(output_tensor_list, dim=0), seq_lens)
        # TODO: 都支持softmax
        if self.config.get("use_multi"):
            pred_output = ClassifierOutputData(seq_unpacked)
        else:
            pred_output = ClassifierOutputData(F.log_softmax(seq_unpacked, dim=-1))
        return pred_output


class MLPClassifier(BaseClassifier):
    """
    Decoder structure based on MLP.
    """
    def __init__(self, **config):
        """ Construction function for Decoder.

        Args:
            config (dict):
                use_slot (bool): whether to classify slot label.
                use_intent (bool): whether to classify intent label.
                mlp (List):
                
                    - _model_target_: torch.nn.Linear
                    
                    in_features (int): input feature dim
                      
                    out_features (int): output feature dim
                      
                    -  _model_target_: torch.nn.LeakyReLU
                    
                    negative_slope: 0.2
                    
                    - ...
        """
        super(MLPClassifier, self).__init__(**config)
        self.config = config
        for i, x in enumerate(config["mlp"]):
            if isinstance(x.get("in_features"), str):
                config["mlp"][i]["in_features"] = self.config[x["in_features"][1:-1]]
            if isinstance(x.get("out_features"), str):
                config["mlp"][i]["out_features"] = self.config[x["out_features"][1:-1]]
        mlp = [instantiate(x) for x in config["mlp"]]
        self.seq = nn.Sequential(*mlp)


    def forward(self, hidden: HiddenData):
        if self.config.get("use_intent"):
            res = self.seq(hidden.intent_hidden)
        else:
            res = self.seq(hidden.slot_hidden)
        return ClassifierOutputData(res)
