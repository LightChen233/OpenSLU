import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from common.utils import HiddenData, ClassifierOutputData
from model.decoder.interaction import BaseInteraction


class LSTMEncoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(LSTMEncoder, self).__init__()

        # Parameter recording.
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim // 2
        self.__dropout_rate = dropout_rate

        # Network attributes.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=self.__embedding_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=self.__dropout_rate,
            num_layers=1
        )

    def forward(self, embedded_text, seq_lens):
        """ Forward process for LSTM Encoder.

        (batch_size, max_sent_len)
        -> (batch_size, max_sent_len, word_dim)
        -> (batch_size, max_sent_len, hidden_dim)

        :param embedded_text: padded and embedded input text.
        :param seq_lens: is the length of original input text.
        :return: is encoded word hidden vectors.
        """

        # Padded_text should be instance of LongTensor.
        dropout_text = self.__dropout_layer(embedded_text)

        # Pack and Pad process for input of variable length.
        packed_text = pack_padded_sequence(dropout_text, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        lstm_hiddens, (h_last, c_last) = self.__lstm_layer(packed_text)
        padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)

        return padded_hiddens


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        B, N = h.size()[0], h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1,
                                                                                                   2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, nlayers=2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.nheads = nheads
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        if self.nlayers > 2:
            for i in range(self.nlayers - 2):
                for j in range(self.nheads):
                    self.add_module('attention_{}_{}'.format(i + 1, j),
                                    GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True))

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        input = x
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        if self.nlayers > 2:
            for i in range(self.nlayers - 2):
                temp = []
                x = F.dropout(x, self.dropout, training=self.training)
                cur_input = x
                for j in range(self.nheads):
                    temp.append(self.__getattr__('attention_{}_{}'.format(i + 1, j))(x, adj))
                x = torch.cat(temp, dim=2) + cur_input
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x + input


def normalize_adj(mx):
    """
    Row-normalize matrix  D^{-1}A
    torch.diag_embed: https://github.com/pytorch/pytorch/pull/12447
    """
    mx = mx.float()
    rowsum = mx.sum(2)
    r_inv = torch.pow(rowsum, -1)
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag_embed(r_inv, 0)
    mx = r_mat_inv.matmul(mx)
    return mx


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
