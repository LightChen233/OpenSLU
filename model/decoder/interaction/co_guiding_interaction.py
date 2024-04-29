
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.module_utils.common_utils import LSTMEncoder, MultiHeadAttention
from common.utils import HiddenData
from model.decoder.interaction.base_interaction import BaseInteraction


class Co_Guiding_GAT(nn.Module):
    def __init__(self, dim_a, dim_b, dropout_rate, n_heads, n_layer, residual = True):
        super(Co_Guiding_GAT, self).__init__()
        self.n_layer = n_layer
        self.residual = residual
        self.a2a_att = nn.ModuleList([MultiHeadAttention(n_heads, dim_a, dim_a//n_heads, dim_a//n_heads, dropout_rate) for i in range(n_layer)])
        self.b2b_att = nn.ModuleList([MultiHeadAttention(n_heads, dim_b, dim_b//n_heads, dim_b//n_heads, dropout_rate) for i in range(n_layer)])
        self.a2b_att = nn.ModuleList([MultiHeadAttention(n_heads, dim_b, dim_a//n_heads, dim_a//n_heads, dropout_rate) for i in range(n_layer)])
        self.b2a_att = nn.ModuleList([MultiHeadAttention(n_heads, dim_a, dim_b//n_heads, dim_b//n_heads, dropout_rate) for i in range(n_layer)])
    
    def forward(self, h_a, h_b, adj_a = None, adj_b = None, adj_ab = None, adj_ba = None):
        input_a = h_a
        input_b = h_b
        for i in range(self.n_layer):
            a2a_out = self.a2a_att[i](h_a, h_a, h_a, adj_a)
            b2b_out = self.b2b_att[i](h_b, h_b, h_b, adj_b)
            a2b_out = self.a2b_att[i](h_b, h_a, h_a, adj_ab)
            b2a_out = self.b2a_att[i](h_a, h_b, h_b, adj_ba)
            h_a = F.relu(a2a_out + b2a_out)
            h_b = F.relu(b2b_out + a2b_out)
        if self.residual:
            return input_a + h_a, input_b  + h_b
        else:
            return h_a, h_b



class CoGuidingInteraction(BaseInteraction):

    def __init__(self, **config):
        super(BaseInteraction, self).__init__()
        self.__num_slot = config["slot_label_num"]
        self.__num_intent = config["intent_label_num"]

        # Initialize an Decoder object for intent.
        self.__intent_decoder = nn.Sequential(
            nn.Linear(config["hidden_dim"],
                      config["hidden_dim"]),
            nn.LeakyReLU(config["negative_slope"]),
            nn.Linear(config["hidden_dim"], self.__num_intent),
        )
        self.__slot_decoder = nn.Sequential(
            nn.Linear(config["hidden_dim"],
                      config["hidden_dim"]),
            nn.LeakyReLU(config["negative_slope"]),
            nn.Linear(config["hidden_dim"], self.__num_slot),
        )

        self.__intent_embedding = nn.Parameter(
            torch.FloatTensor(self.__num_intent, config["hidden_dim"]))
        nn.init.normal_(self.__intent_embedding.data)
        self.__slot_embedding = nn.Parameter(
            torch.FloatTensor(self.__num_slot, config["hidden_dim"]))
        self.__slot_lstm = LSTMEncoder(
            config["input_dim"],
            config["hidden_dim"],
            config["dropout_rate"]
        )
        self.__intent_lstm = LSTMEncoder(
            config["input_dim"],
            config["hidden_dim"],
            config["dropout_rate"]
        )
        
        self.__slot_hgat_lstm = nn.ModuleList([LSTMEncoder(
            config["hidden_dim"]+ self.__num_intent,
            config["hidden_dim"],
            config["dropout_rate"]
        ) for i in range(config["layer_num"])])
        
        self.__slot_hgat= nn.ModuleList([slot_hgat(
            config["hidden_dim"],
            config["gat_dropout_rate"], 
            config["n_heads"], 
            config["n_layers_decoder_global"]) for i in range(config["layer_num"])])
        self.__intent_hgat= nn.ModuleList([intent_hgat(
            config["hidden_dim"],
            config["gat_dropout_rate"], 
            config["n_heads"], 
            config["n_layers_decoder_global"]) for i in range(config["layer_num"])])
        self.config = config
    
    def generate_self_local_adj(self, seq_len, batch, window):
        adj = torch.cat([torch.eye(torch.max(seq_len)).unsqueeze(0) for i in range(batch)])
        for i in range(batch):
            for j in range(seq_len[i]):
                adj[i, j, max(0, j - window):min(seq_len[i], j + window + 1)] = 1.
        
        return adj

    def generate_slot_hgat_adj(self, seq_len, index, batch):
        global_intent_idx = [[] for i in range(batch)]
        #global_slot_idx = [[] for i in range(batch)]
        for item in index:
            global_intent_idx[item[0]].append(item[1])

        #for i, len in enumerate(seq_len):
         #   global_slot_idx[i].extend(list(range(self.__num_intent, self.__num_intent + len)))

        intent2slot_adj = torch.cat([torch.zeros(torch.max(seq_len), self.__num_intent).unsqueeze(0) for i in range(batch)])
        intent_adj = torch.cat([torch.zeros(self.__num_intent, self.__num_intent).unsqueeze(0) for i in range(batch)])
        slot2intent_adj = torch.cat([torch.zeros(self.__num_intent, torch.max(seq_len)).unsqueeze(0) for i in range(batch)])
        for i in range(batch):
            for j in range(seq_len[i]):
                intent2slot_adj[i, j, global_intent_idx[i]] = 1.
            for j in range(self.__num_intent):
                intent_adj[i, j, global_intent_idx[i]] = 1.
                slot2intent_adj[i, j, :seq_len[i]] = 1
        return intent_adj, intent2slot_adj, slot2intent_adj

    def forward(self, encode_hidden: HiddenData, **kwargs):
        device =encode_hidden.get_intent_hidden_state().device
        seq_lens = encode_hidden.inputs.attention_mask.sum(-1).cpu()
        self_local_adj = self.generate_self_local_adj(seq_lens,
                                                      len(seq_lens),
                                                      self.config["slot_graph_window"]).to(device)
        intent_lstm_out = self.__intent_lstm(encode_hidden.get_slot_hidden_state(), seq_lens)
        intent_lstm_out = F.dropout(intent_lstm_out, p=self.config["dropout_rate"], training=self.training)
        intent_logits = self.__intent_decoder(intent_lstm_out)
        
        slot_lstm_out = self.__slot_lstm(encode_hidden.get_slot_hidden_state(), seq_lens)
        slot_lstm_out = F.dropout(slot_lstm_out, p=self.config["dropout_rate"], training=self.training)
        slot_logits = self.__slot_decoder(slot_lstm_out)
       
        seq_lens_tensor = torch.tensor(seq_lens).to(device)

        slot_logit_list, intent_logit_list = [], []
        slot_logit_list.append(slot_logits)
        intent_logit_list.append(intent_logits)
        h_slot = slot_lstm_out
        h_intent = intent_lstm_out
        for i in range(self.config["layer_num"]):
            intent_index_sum = torch.cat(
            [
                torch.sum(torch.sigmoid(intent_logits[i, 0:seq_lens[i], :]) > self.config["threshold"], dim=0).unsqueeze(0)
                for i in range(len(seq_lens))
            ], dim=0)
            intent_index = (intent_index_sum > (seq_lens_tensor // 2).unsqueeze(1)).nonzero()
            intent_adj, intent2slot_adj, slot2intent_adj = self.generate_slot_hgat_adj(seq_lens, intent_index, len(seq_lens))
            intent_adj, intent2slot_adj, slot2intent_adj = intent_adj.to(device), intent2slot_adj.to(device), slot2intent_adj.to(device)
            slot_index = torch.argmax(slot_logits, dim = -1)
            slot_one_hot = F.one_hot(slot_index, num_classes = self.__num_slot)
            slot_embedding = torch.matmul(slot_one_hot.float(), self.__slot_embedding) 
           
            slot_lstm_out = self.__slot_hgat_lstm[i](torch.cat([h_slot, intent_logits], dim=-1), seq_lens)
            slot_lstm_out = F.dropout(slot_lstm_out, p=self.config["dropout_rate"], training=self.training)       
            
            h_slot = self.__slot_hgat[i](
                slot_lstm_out, seq_lens, 
                adj_i=intent_adj,
                adj_s=self_local_adj,
                adj_is = intent2slot_adj,
                adj_si = slot2intent_adj,
                intent_embedding=self.__intent_embedding)
            
            h_intent = self.__intent_hgat[i](
                h_intent,
                adj_i = self_local_adj,
                adj_s = self_local_adj,
                adj_is = self_local_adj,
                adj_si = self_local_adj,
                slot_embedding=slot_embedding)
        encode_hidden.update_intent_hidden_state(torch.cat([intent_lstm_out, h_intent], dim = -1))
        encode_hidden.update_slot_hidden_state(torch.cat([slot_lstm_out, h_slot], dim = -1))
        return encode_hidden

class slot_hgat(nn.Module):

    def __init__(self, hidden_dim, gat_dropout_rate, n_heads, n_layers_decoder_global):

        super(slot_hgat, self).__init__()

        self.__global_graph = Co_Guiding_GAT(
            hidden_dim, hidden_dim,
            gat_dropout_rate, n_heads,
            n_layers_decoder_global)


    def forward(self, encoded_hiddens, seq_lens, adj_i, adj_s, adj_is, adj_si, intent_embedding):

        batch = len(seq_lens)
        intent_in = intent_embedding.unsqueeze(0).repeat(batch, 1, 1)
        _,slot_out = self.__global_graph(intent_in, encoded_hiddens,adj_i, adj_s, adj_is, adj_si)

        return slot_out

class intent_hgat(nn.Module):

    def __init__(self, hidden_dim, gat_dropout_rate, n_heads, n_layers_decoder_global):

        super(intent_hgat, self).__init__()
        self.__global_graph = Co_Guiding_GAT(hidden_dim, hidden_dim, gat_dropout_rate, n_heads, n_layers_decoder_global)

    def forward(self, h_intent, adj_i, adj_s, adj_is, adj_si, slot_embedding):

        intent_out, _ = self.__global_graph(h_intent, slot_embedding, adj_i, adj_s, adj_is, adj_si)
        
        return intent_out


