'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-02-22 13:29:48
LastEditTime: 2023-05-01 18:22:05
Description: 

'''
import torch
import torch.nn as nn
from common.module_utils.common_utils import LSTMEncoder
from common.module_utils.graph_utils import GAT, normalize_adj

from common.utils import HiddenData, ClassifierOutputData
from model.decoder.interaction import BaseInteraction

class GLGINInteraction(BaseInteraction):
    def __init__(self, **config):
        super().__init__(**config)
        self.intent_embedding = nn.Parameter(
            torch.FloatTensor(self.config["intent_label_num"], self.config["intent_embedding_dim"]))  # 191, 32
        nn.init.normal_(self.intent_embedding.data)
        self.adj = None
        self.__slot_lstm = LSTMEncoder(
            self.config["input_dim"] + self.config["intent_label_num"],
            config["output_dim"],
            config["dropout_rate"]
        )
        self.__slot_graph = GAT(
            config["output_dim"],
            config["hidden_dim"],
            config["output_dim"],
            config["dropout_rate"],
            config["alpha"],
            config["num_heads"],
            config["num_layers"])

        self.__global_graph = GAT(
            config["output_dim"],
            config["hidden_dim"],
            config["output_dim"],
            config["dropout_rate"],
            config["alpha"],
            config["num_heads"],
            config["num_layers"])

    def generate_global_adj_gat(self, seq_len, index, batch, window):
        global_intent_idx = [[] for i in range(batch)]
        global_slot_idx = [[] for i in range(batch)]
        for item in index:
            global_intent_idx[item[0]].append(item[1])

        for i, len in enumerate(seq_len):
            global_slot_idx[i].extend(
                list(range(self.config["intent_label_num"], self.config["intent_label_num"] + len)))

        adj = torch.cat([torch.eye(self.config["intent_label_num"] + max(seq_len)).unsqueeze(0) for i in range(batch)])
        for i in range(batch):
            for j in global_intent_idx[i]:
                adj[i, j, global_slot_idx[i]] = 1.
                adj[i, j, global_intent_idx[i]] = 1.
            for j in global_slot_idx[i]:
                adj[i, j, global_intent_idx[i]] = 1.

        for i in range(batch):
            for j in range(self.config["intent_label_num"], self.config["intent_label_num"] + seq_len[i]):
                adj[i, j, max(self.config["intent_label_num"], j - window):min(seq_len[i] + self.config["intent_label_num"], j + window + 1)] = 1.

        if self.config["row_normalized"]:
            adj = normalize_adj(adj)
        adj = adj.to(self.intent_embedding.device)
        return adj

    def generate_slot_adj_gat(self, seq_len, batch, window):
        slot_idx_ = [[] for i in range(batch)]
        adj = torch.cat([torch.eye(max(seq_len)).unsqueeze(0) for i in range(batch)])
        for i in range(batch):
            for j in range(seq_len[i]):
                adj[i, j, max(0, j - window):min(seq_len[i], j + window + 1)] = 1.
        if self.config["row_normalized"]:
            adj = normalize_adj(adj)
        adj = adj.to(self.intent_embedding.device)
        return adj

    def forward(self, encode_hidden: HiddenData, pred_intent: ClassifierOutputData = None, intent_index=None):
        seq_lens = encode_hidden.inputs.attention_mask.sum(-1)
        slot_lstm_out = self.__slot_lstm(torch.cat([encode_hidden.slot_hidden, pred_intent.classifier_output], dim=-1),
                                         seq_lens)
        global_adj = self.generate_global_adj_gat(seq_lens, intent_index, len(seq_lens),
                                                  self.config["slot_graph_window"])
        slot_adj = self.generate_slot_adj_gat(seq_lens, len(seq_lens), self.config["slot_graph_window"])
        batch = len(seq_lens)
        slot_graph_out = self.__slot_graph(slot_lstm_out, slot_adj)
        intent_in = self.intent_embedding.unsqueeze(0).repeat(batch, 1, 1)
        global_graph_in = torch.cat([intent_in, slot_graph_out], dim=1)
        encode_hidden.update_slot_hidden_state(self.__global_graph(global_graph_in, global_adj))
        return encode_hidden
