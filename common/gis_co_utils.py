'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-05-01 00:26:15
LastEditTime: 2023-06-03 18:00:34
Description: 

'''
import json, os
import numpy
import torch
import pandas
import scipy.sparse as sp
from datasets import load_dataset


def get_graph_adj(data_dir, save_dir, intent_dict, slot_dict):
    file = data_dir
    file_label = file + "/label"
    file_seq_out = file + "/seq.out"
    label = []
    seq_out = []
    
    num_intent = len(intent_dict)
    num_slot = len(slot_dict)
    stat = [[0] * num_slot for _ in range(num_intent)]
    intent_stat = [0] * num_intent
    slot_stat = [0] * num_slot
    with open(file_label) as r1, open(file_seq_out) as r3:
        for l1 in r1:
            l3 = r3.readline()
            label.append(l1.strip())
            seq_out.append(l3.strip())
    
    for la, so in zip(label, seq_out):
        if '#' not in la:
            intent_id = int(intent_dict[la])
            intent_stat[intent_id] += 1
            slot_list = so.split(' ')
            for slot in slot_list:
                if slot in slot_dict.keys():
                    slot_id = int(slot_dict[slot])
                    slot_stat[slot_id] += 1
                    stat[intent_id][slot_id] += 1
        elif '#' in la:
            intent_list = la.split('#')
            for i in intent_list:
                intent_id = int(intent_dict[i])
                intent_stat[intent_id] += 1
                slot_list = so.split(' ')
                for slot in slot_list:
                    if slot in slot_dict.keys():
                        slot_id = int(slot_dict[slot])
                        slot_stat[slot_id] += 1
                        stat[intent_id][slot_id] += 1
        else:
            print('error!!!')
    
    
    
    data = numpy.array(stat)
    data_intent = numpy.array(intent_stat)
    data_slot = numpy.array(slot_stat)
    data = numpy.delete(data, [0], axis=1)
    data_slot = numpy.delete(data_slot, [0])

    o_1 = numpy.array([0])
    o_2 = numpy.array([[0] * num_intent])
    data_slot = numpy.insert(data_slot, 0, o_1)
    data = numpy.insert(data, 0, o_2, axis=1)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    numpy.savetxt(os.path.join(save_dir, "data_intent.txt"), data_intent)
    numpy.savetxt(os.path.join(save_dir, "data_slot.txt"), data_slot)
    numpy.savetxt(os.path.join(save_dir, "data.txt"), data)

    df = pandas.DataFrame(data)
    df_intent = pandas.Series(data_intent)
    df_slot = pandas.Series(data_slot)
    intent_2_slot = df.div(df_intent + 1e-10, axis=0)
    slot_2_intent = df.div(df_slot + 1e-10, axis=1)
    
    A_right_up = intent_2_slot.values
    A_left_down = slot_2_intent.values.T
    A_left_up = numpy.array([[0] * num_intent for _ in range(num_intent)])
    A_right_down = numpy.array([[0] * (num_slot) for _ in range(num_slot)])
    A_left = numpy.concatenate((A_left_up, A_left_down), axis=0)
    A_right = numpy.concatenate((A_right_up, A_right_down), axis=0)
    A = numpy.concatenate((A_left, A_right), axis=1)
    
    A_eye = numpy.eye(num_intent + num_slot)
    A = A + A_eye
    numpy.savetxt(os.path.join(save_dir, "graph_adj.txt"), A)
    adj = torch.from_numpy(A).float()
    adj = sp.coo_matrix(adj)
    rowsum = numpy.array(adj.sum(1))
    d_inv_sqrt = numpy.power(rowsum, -0.5).flatten()
    d_inv_sqrt[numpy.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    output_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    output_adj = torch.FloatTensor(numpy.array(output_adj.todense()))
    numpy.savetxt(os.path.join(save_dir, "graph_output_adj.txt"), numpy.array(output_adj))

    return output_adj

def parse_cocurrence(dataset_name, dataset_path, split):
    def get_data(dataset_name, dataset_path, split):
        if dataset_name is not None and dataset_path is None:
            return load_dataset("LightChen2333/OpenSLU", dataset_name, split=split)
        elif dataset_path is not None:
            data_file = dataset_path
            data_dict = {"text": [], "slot": [], "intent":[]}
            with open(data_file, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    if len(row["text"]) == len(row["slot"]):
                        data_dict["text"].append(row["text"])
                        data_dict["slot"].append(row["slot"])
                        data_dict["intent"].append(row["intent"])
                    else:
                        print("Error Input: ", row)
            return data_dict
        else:
            return {}
    res = get_data(dataset_name, dataset_path, split)
    if not os.path.exists("common/co_occurence/"+ dataset_name):
        os.mkdir("common/co_occurence/"+ dataset_name)
    with open("common/co_occurence/"+ dataset_name + "/label", "w", encoding="utf8") as f:
        for x in res["intent"]:
            f.write(x+"\n")
    with open("common/co_occurence/"+ dataset_name + "/seq.out", "w", encoding="utf8") as f:
        for x in res["slot"]:
            f.write(" ".join(x)+"\n")
    
if __name__ == "__main__":
    parse_cocurrence(dataset_name="atis", dataset_path=None, split="train")
    parse_cocurrence(dataset_name="snips", dataset_path=None, split="train")