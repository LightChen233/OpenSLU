'''
Author: Qiguang Chen
Date: 2023-01-11 10:39:26
LastEditors: Qiguang Chen
LastEditTime: 2023-02-20 09:30:56
Description: log manager

'''
import datetime
import json
import os
import time
from common.config import Config
import logging
import colorlog

def mkdirs(dir_names):
    for dir_name in dir_names:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)



class Logger():
    """ logging infomation by [wandb, fitlog, local file]
    """
    def __init__(self, 
                 logger_type: str,
                 logger_name: str,
                 logging_level="INFO",
                 start_time='',
                 accelerator=None):
        """ create logger

        Args:
            logger_type (str): support type = ["wandb", "fitlog", "local"]
            logger_name (str): logger name, means project name in wandb, and logging file name
            logging_level (str, optional): logging level. Defaults to "INFO".
            start_time (str, optional): start time string. Defaults to ''.
        """
        self.logger_type = logger_type
        self.output_dir = "logs/" + logger_name + "/" + start_time
        self.accelerator = accelerator
        self.logger_name = logger_name
        if accelerator is not None:
            from accelerate.logging import get_logger
            self.logging = get_logger(logger_name)
        else:
            if self.logger_type == "wandb":
                import wandb
                self.logger = wandb
                mkdirs(["logs", "logs/" + logger_name, self.output_dir])
                self.logger.init(project=logger_name)
            elif self.logger_type == "fitlog":
                import fitlog
                self.logger = fitlog
                mkdirs(["logs", "logs/" + logger_name, self.output_dir])
                self.logger.set_log_dir("logs/" + logger_name)
            else:
                mkdirs(["logs", "logs/" + logger_name, self.output_dir])
                self.config_file = os.path.join(self.output_dir, "config.jsonl")
                with open(self.config_file, "w", encoding="utf8") as f:
                    print(f"Config will be written to {self.config_file}")

                self.loss_file = os.path.join(self.output_dir, "loss.jsonl")
                with open(self.loss_file, "w", encoding="utf8") as f:
                    print(f"Loss Result will be written to {self.loss_file}")

                self.metric_file = os.path.join(self.output_dir, "metric.jsonl")
                with open(self.metric_file, "w", encoding="utf8") as f:
                    print(f"Metric Result will be written to {self.metric_file}")

                self.other_log_file = os.path.join(self.output_dir, "other_log.jsonl")
                with open(self.other_log_file, "w", encoding="utf8") as f:
                    print(f"Other Log Result will be written to {self.other_log_file}")
            
            self.logging = self._get_logging_logger(logging_level)

    def _get_logging_logger(self, level="INFO"):
        log_colors_config = {
            'DEBUG': 'cyan',
            'INFO': 'blue',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
 
        logger = logging.getLogger()
        logger.setLevel(level)

        log_path = os.path.join(self.output_dir, "log.log")
 
        if not logger.handlers:  
            sh = logging.StreamHandler()
            fh = logging.FileHandler(filename=log_path, mode='a', encoding="utf-8")
            fmt = logging.Formatter(
                fmt='[%(levelname)s - %(asctime)s]\t%(message)s',
                datefmt='%m/%d/%Y %I:%M:%S %p')
 
            sh_fmt = colorlog.ColoredFormatter(
                fmt='%(log_color)s[%(levelname)s - %(asctime)s]\t%(message)s',
                datefmt='%m/%d/%Y %I:%M:%S %p',
                log_colors=log_colors_config)
            sh.setFormatter(fmt=sh_fmt)
            fh.setFormatter(fmt=fmt)
            logger.addHandler(sh)
            logger.addHandler(fh)
        return logger
    
    def set_config(self, config: Config):
        """save config

        Args:
            config (Config): configuration object to save
        """
        if self.accelerator is not None:
            self.accelerator.init_trackers(self.logger_name, config=config)
        elif self.logger_type == "wandb":
            self.logger.config.update(config)
        elif self.logger_type == "fitlog":
            self.logger.add_hyper(config)
        else:
            with open(self.config_file, "a", encoding="utf8") as f:
                f.write(json.dumps(config) + "\n")

    def log(self, data, step=0):
        """log data and step

        Args:
            data (Any): data to log
            step (int, optional): step num. Defaults to 0.
        """
        if self.accelerator is not None:
            self.accelerator.log(data, step=0)
        elif self.logger_type == "wandb":
            self.logger.log(data, step=step)
        elif self.logger_type == "fitlog":
            self.logger.add_other({"data": data, "step": step})
        else:
            with open(self.other_log_file, "a", encoding="utf8") as f:
                f.write(json.dumps({"data": data, "step": step}) + "\n")

    def log_metric(self, metric, metric_split="dev", step=0):
        """log metric

        Args:
            metric (Any): metric
            metric_split (str, optional): dataset split. Defaults to 'dev'.
            step (int, optional): step num. Defaults to 0.
        """
        if self.accelerator is not None:
            self.accelerator.log({metric_split: metric}, step=step)
        elif self.logger_type == "wandb":
            self.logger.log({metric_split: metric}, step=step)
        elif self.logger_type == "fitlog":
            self.logger.add_metric({metric_split: metric}, step=step)
        else:
            with open(self.metric_file, "a", encoding="utf8") as f:
                f.write(json.dumps({metric_split: metric, "step": step}) + "\n")

    def log_loss(self, loss, loss_name="Loss", step=0):
        """log loss

        Args:
            loss (Any): loss
            loss_name (str, optional): loss description. Defaults to 'Loss'.
            step (int, optional): step num. Defaults to 0.
        """
        if self.accelerator is not None:
            self.accelerator.log({loss_name: loss}, step=step)
        elif self.logger_type == "wandb":
            self.logger.log({loss_name: loss}, step=step)
        elif self.logger_type == "fitlog":
            self.logger.add_loss(loss, name=loss_name, step=step)
        else:
            with open(self.loss_file, "a", encoding="utf8") as f:
                f.write(json.dumps({loss_name: loss, "step": step}) + "\n")

    def finish(self):
        """finish logging
        """
        if self.logger_type == "fitlog":
            self.logger.finish()

    def info(self, message:str):
        """ Log a message with severity 'INFO' in local file / console.
        
        Args:
            message (str): message to log
        """
        self.logging.info(message)

    def warning(self, message):
        """ Log a message with severity 'WARNING' in local file / console.
        
        Args:
            message (str): message to log
        """
        self.logging.warning(message)

    def error(self, message):
        """ Log a message with severity 'ERROR' in local file / console.
        
        Args:
            message (str): message to log
        """
        self.logging.error(message)

    def debug(self, message):
        """ Log a message with severity 'DEBUG' in local file / console.
        
        Args:
            message (str): message to log
        """
        self.logging.debug(message)

    def critical(self, message):
        self.logging.critical(message)
