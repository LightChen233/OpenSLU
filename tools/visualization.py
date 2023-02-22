'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-01-23 17:26:47
LastEditTime: 2023-02-14 20:07:02
Description: 

'''
import argparse
import os
import signal
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from gradio import networking
from common.utils import load_yaml, str2bool
import json
import threading

from flask import Flask, request, render_template, render_template_string


def get_example(start, end, predict_data_file_path):
    data_list = []
    with open(predict_data_file_path, "r", encoding="utf8") as f1:
        for index, line1 in enumerate(f1):
            if index < start:
                continue
            if index > end:
                break
            line1 = json.loads(line1.strip())
            obj = {"text": line1["text"]}
            obj["intent"] = [{"intent": line1["golden_intent"],
                              "pred_intent": line1["pred_intent"]}]
            obj["slot"] = [{"text": t, "pred_slot": ps, "slot": s} for t, s, ps in zip(
                line1["text"], line1["pred_slot"], line1["golden_slot"])]
            data_list.append(obj)
    return data_list


def analysis(predict_data_file_path):
    intent_dict = {}
    slot_dict = {}
    sample_num = 0
    with open(predict_data_file_path, "r", encoding="utf8") as f1:
        for index, line1 in enumerate(f1):
            sample_num += 1
            line1 = json.loads(line1.strip())
            for s, ps in zip(line1["golden_slot"], line1["pred_slot"]):
                if s not in slot_dict:
                    slot_dict[s] = {"_error_": 0, "_total_": 0}
                if s != ps:
                    slot_dict[s]["_error_"] += 1
                    if ps not in slot_dict[s]:
                        slot_dict[s][ps] = 0
                    slot_dict[s][ps] += 1
                slot_dict[s]["_total_"] += 1
            for i, pi in zip([line1["golden_intent"]], [line1["pred_intent"]]):
                if i not in intent_dict:
                    intent_dict[i] = {"_error_": 0, "_total_": 0}
                if i != pi:
                    intent_dict[i]["_error_"] += 1
                    if pi not in intent_dict[i]:
                        intent_dict[i][pi] = 0
                    intent_dict[i][pi] += 1
                intent_dict[i]["_total_"] += 1
    intent_dict_list = [{"value": intent_dict[name]["_error_"], "name": name} for name in intent_dict]
    
    for intent in intent_dict_list:
        temp_intent = sorted(
            intent_dict[intent["name"]].items(), key=lambda d: d[1], reverse=True)
        # [:7]
        temp_intent = [[key, value] for key, value in temp_intent]
        intent_dict[intent["name"]] = temp_intent
    slot_dict_list = [{"value": slot_dict[name]["_error_"], "name": name} for name in slot_dict]
    
    for slot in slot_dict_list:
        temp_slot = sorted(
            slot_dict[slot["name"]].items(), key=lambda d: d[1], reverse=True)
        temp_slot = [[key, value] for key, value in temp_slot]
        slot_dict[slot["name"]] = temp_slot
    return intent_dict_list, slot_dict_list, intent_dict, slot_dict, sample_num

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', '-cp', type=str, default="config/visual.yaml")
parser.add_argument('--output_path', '-op', type=str, default=None)
parser.add_argument('--push_to_public', '-p', type=str2bool, nargs='?',
                        const=True, default=None,
                        help="Push to public network.(Higher priority than config file)")
args = parser.parse_args()
button_html = ""
config = load_yaml(args.config_path)
if args.output_path is not None:
    config["output_path"] = args.output_path
if args.push_to_public is not None:
    config["is_push_to_public"] = args.push_to_public
intent_dict_list, slot_dict_list, intent_dict, slot_dict, sample_num = analysis(config["output_path"])
PAGE_SIZE = config["page-size"]
PAGE_NUM = int(sample_num / PAGE_SIZE) + 1

app = Flask(__name__, template_folder="static//template")


@app.route("/")
def hello():
    page = request.args.get('page')
    if page is None:
        page = 0
    page = int(page) if int(page) >= 0 else 0
    init_index = page*PAGE_SIZE
    examples = get_example(init_index, init_index +
                           PAGE_SIZE - 1, config["output_path"])
    return render_template('visualization.html',
                           examples=examples,
                           intent_dict_list=intent_dict_list,
                           slot_dict_list=slot_dict_list,
                           intent_dict=intent_dict,
                           slot_dict=slot_dict,
                           page=page)

thread_lock_1 = False



    
class PushToPublicThread():
    def __init__(self, config) -> None:
        self.thread = threading.Thread(target=self.push_to_public, args=(config,))
        self.thread_lock_2 = False
        self.thread.daemon = True
    
    def start(self):
        self.thread.start()
    
    def push_to_public(self, config):
        print("Push visualization results to public by Gradio....")
        print("Push to URL: ", networking.setup_tunnel(config["host"], str(config["port"])))
        print("This share link expires in 72 hours. And do not close this process for public sharing.")
        while not self.thread_lock_2:
            continue
    
    def exit(self, signum, frame):
        self.thread_lock_2 = True
        print("Exit..")
        os._exit(0)
        # exit()
if __name__ == '__main__':
    
    if config["is_push_to_public"]:
        
        thread_1 = threading.Thread(target=lambda: app.run(
            config["host"], config["port"]))
        thread_1.start()
        thread_2 = PushToPublicThread(config)
        signal.signal(signal.SIGINT, thread_2.exit)
        signal.signal(signal.SIGTERM, thread_2.exit)
        thread_2.start()
        while True:
            time.sleep(1)
    else:
        app.run(config["host"], config["port"])
