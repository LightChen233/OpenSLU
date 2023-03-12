'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-02-12 22:23:58
LastEditTime: 2023-03-08 21:56:45
Description: 

'''
import json
import os
import queue
import shutil
import torch
import dill
from common import utils

def safe_mkdir(path, accelerator=None):
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except OSError:
        pass


class Saver():
    def __init__(self, config, start_time=None, accelerator=None) -> None:
        self.config = config
        self.accelerator = accelerator
        if self.config.get("save_dir"):
            self.model_save_dir = self.config["save_dir"]
        else:
            safe_mkdir("save/")
            self.model_save_dir = "save/" + start_time
        safe_mkdir(self.model_save_dir)
        save_mode = config.get("save_mode")
        self.save_mode = save_mode if save_mode is not None else "save-by-eval"
        
        max_save_num = self.config.get("max_save_num")
        self.max_save_num = max_save_num if max_save_num is not None else 1
        self.save_pool = queue.Queue(maxsize=max_save_num)
    
    def save_tokenizer(self, tokenizer):
        with open(os.path.join(self.model_save_dir, "tokenizer.pkl"), 'wb') as f:
            dill.dump(tokenizer, f)
    
    def save_label(self, intent_list, slot_list):
        utils.save_json(os.path.join(self.model_save_dir, "label.json"), {"intent": intent_list, "slot": slot_list})
    
    
    def use_validation(self):
        return self.save_mode != "save-by-step"
    
    def save_model(self, model, train_state, accelerator=None):
        step = train_state["step"]
        if self.max_save_num != 1:
            model_save_dir =os.path.join(self.model_save_dir, str(step))
            try:
                if self.save_pool.full():
                    delete_dir = self.save_pool.get()
                    shutil.rmtree(delete_dir)
                    self.save_pool.put(model_save_dir)
                else:
                    self.save_pool.put(model_save_dir)
                if not os.path.exists(model_save_dir):
                    os.mkdir(model_save_dir)
            except OSError:
                pass
        else:
            model_save_dir = self.model_save_dir
            safe_mkdir(model_save_dir)
        if accelerator is None:
            torch.save(model, os.path.join(model_save_dir, "model.pkl"))
            torch.save(train_state, os.path.join(model_save_dir, "train_state.pkl"), pickle_module=dill)
        else:
            # accelerator.wait_for_everyone()
            # unwrapped_model = accelerator.unwrap_model(model)
            # accelerator.save(unwrapped_model, os.path.join(model_save_dir, "model.pkl"))
            accelerator.save_state(output_dir=model_save_dir)
    
    def auto_save_step(self, model, train_state, accelerator=None):
        step = train_state["step"]
        if self.save_mode == "save-by-step" and step % self.config.get("save_step")==0 and step != 0:
            self.save_model(model, train_state, accelerator)
            return True
        else:
            return False
        
    
    def save_output(self, outputs, dataset):
        outputs.save(self.model_save_dir, dataset)