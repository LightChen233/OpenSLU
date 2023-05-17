'''
Author: Qiguang Chen
Date: 2023-01-11 10:39:26
LastEditors: Qiguang Chen
LastEditTime: 2023-05-01 23:02:05
Description: non-pretrained encoder model

'''
import math

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from common.utils import HiddenData, InputData
from model.encoder.base_encoder import BaseEncoder

class NonPretrainedEncoder(BaseEncoder):
    """
    Encoder structure based on bidirectional LSTM and self-attention.
    """

    def __init__(self, **config):
        """ init non-pretrained encoder
        
        Args:
            config (dict):
                embedding (dict):
                    dropout_rate        (float): dropout rate.
                    load_embedding_name (str): None if not use pretrained embedding or embedding name like "glove.6B.300d.txt".
                    embedding_matrix    (Tensor, Optional): embedding matrix tensor. Enabled if load_embedding_name is not None. 
                    vocab_size          (str, Optional): vocabulary size. Enabled if load_embedding_name is None.
                lstm (dict):
                    output_dim    (int): lstm output dim.
                    bidirectional (bool): if use BiLSTM or LSTM.
                    layer_num     (int): number of layers.
                    dropout_rate  (float): dropout rate.
                attention (dict, Optional):
                    dropout_rate (float): dropout rate.
                    hidden_dim   (int): attention hidden dim.
                    output_dim   (int): attention output dim.
                unflat_attention (dict, optional): Enabled if attention is not None.
                    dropout_rate (float): dropout rate.
        """
        super(NonPretrainedEncoder, self).__init__()
        self.config = config
        # Embedding Initialization
        embed_config = config["embedding"]
        self.__embedding_dim = embed_config["embedding_dim"]
        if embed_config.get("load_embedding_name") and embed_config.get("embedding_matrix") is not None:
            self.__embedding_layer = nn.Embedding.from_pretrained(embed_config["embedding_matrix"], padding_idx=0)
        else:
            self.__embedding_layer = nn.Embedding(
                embed_config["vocab_size"], self.__embedding_dim
            )
        self.__embedding_dropout_layer = nn.Dropout(embed_config["dropout_rate"])

        # LSTM Initialization
        lstm_config = config["lstm"]
        self.__hidden_size = lstm_config["output_dim"]
        self.__lstm_layer = nn.LSTM(
            input_size=self.__embedding_dim,
            hidden_size=lstm_config["output_dim"] // 2,
            batch_first=True,
            bidirectional=lstm_config["bidirectional"],
            dropout=lstm_config["dropout_rate"],
            num_layers=lstm_config["layer_num"]
        )
        if self.config.get("attention"):
            # Attention Initialization
            att_config = config["attention"]
            self.__attention_dropout_layer = nn.Dropout(att_config["dropout_rate"])
            self.__attention_layer = QKVAttention(
                self.__embedding_dim, self.__embedding_dim, self.__embedding_dim,
                att_config["hidden_dim"], att_config["output_dim"], att_config["dropout_rate"]
            )
            if self.config.get("unflat_attention"):
                unflat_att_config = config["unflat_attention"]
                self.__sentattention = UnflatSelfAttention(
                    lstm_config["output_dim"] + att_config["output_dim"],
                    unflat_att_config["dropout_rate"]
                )

    def forward(self, inputs: InputData):
        """ Forward process for Non-Pretrained Encoder.

        Args:
            inputs: padded input ids, masks.
        Returns:
            encoded hidden vectors.
        """

        # LSTM Encoder
        # Padded_text should be instance of LongTensor.
        embedded_text = self.__embedding_layer(inputs.input_ids)
        dropout_text = self.__embedding_dropout_layer(embedded_text)
        seq_lens = inputs.attention_mask.sum(-1).detach().cpu()
        # Pack and Pad process for input of variable length.
        packed_text = pack_padded_sequence(dropout_text, seq_lens, batch_first=True, enforce_sorted=False)
        lstm_hiddens, (h_last, c_last) = self.__lstm_layer(packed_text)
        padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)

        if self.config.get("attention"):
            # Attention Encoder
            dropout_text = self.__attention_dropout_layer(embedded_text)
            attention_hiddens = self.__attention_layer(
                dropout_text, dropout_text, dropout_text, mask=inputs.attention_mask
            )

            # Attention + LSTM
            hiddens = torch.cat([attention_hiddens, padded_hiddens], dim=-1)
            hidden = HiddenData(None, hiddens)
            if self.config.get("return_with_input"):
                hidden.add_input(inputs)
            if self.config.get("return_sentence_level_hidden"):
                if self.config.get("unflat_attention"):
                    sentence = self.__sentattention(hiddens, seq_lens)
                else:
                    sentence = hiddens[:, 0, :]
                hidden.update_intent_hidden_state(sentence)
        else:
            sentence_hidden = None
            if self.config.get("return_sentence_level_hidden"):
                sentence_hidden = torch.cat((h_last[-1], h_last[-1], c_last[-1], c_last[-2]), dim=-1)
            hidden = HiddenData(sentence_hidden, padded_hiddens)
            if self.config.get("return_with_input"):
                hidden.add_input(inputs)

        return hidden


class QKVAttention(nn.Module):
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        # Record hyper-parameters.
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Declare network structures.
        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
        self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value, mask=None):
        """ The forward propagation of attention.

        Here we require the first dimension of input key
        and value are equal.

        Args:
            input_query: is query tensor, (n, d_q)
            input_key:  is key tensor, (m, d_k)
            input_value:  is value tensor, (m, d_v)
        
        Returns:
            attention based tensor, (n, d_h)
        """

        # Linear transform to fine-tune dimension.
        linear_query = self.__query_layer(input_query)
        linear_key = self.__key_layer(input_key)
        linear_value = self.__value_layer(input_value)

        score_tensor = torch.matmul(
            linear_query,
            linear_key.transpose(-2, -1)
        ) / math.sqrt(self.__hidden_dim)
        if mask is not None:
            attn_mask = einops.repeat((mask == 0), "b l -> b l h", h=score_tensor.shape[-1])
            score_tensor = score_tensor.masked_fill_(attn_mask, -float(1e20))
        score_tensor = F.softmax(score_tensor, dim=-1)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor


class UnflatSelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0.):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, lens):
        batch_size, seq_len, d_feat = inp.size()
        inp = self.dropout(inp)
        scores = self.scorer(inp.contiguous().view(-1, d_feat)).view(batch_size, seq_len)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores.data[i, l:] = -np.inf
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(inp).mul(inp).sum(1)
        return context