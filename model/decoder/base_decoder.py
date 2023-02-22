'''
Author: Qiguang Chen
Date: 2023-01-11 10:39:26
LastEditors: Qiguang Chen
LastEditTime: 2023-01-31 18:22:36
Description: 

'''
from torch import nn

from common.utils import HiddenData, OutputData, InputData


class BaseDecoder(nn.Module):
    """Base class for all decoder module. 
    
    Notice: t is often only necessary to change this module and its sub-modules
    """
    def __init__(self, intent_classifier=None, slot_classifier=None, interaction=None):
        super().__init__()
        self.intent_classifier = intent_classifier
        self.slot_classifier = slot_classifier
        self.interaction = interaction

    def forward(self, hidden: HiddenData):
        """forward

        Args:
            hidden (HiddenData): encoded data

        Returns:
            OutputData: prediction logits 
        """
        if self.interaction is not None:
            hidden = self.interaction(hidden)
        intent = None
        slot = None
        if self.intent_classifier is not None:
            intent = self.intent_classifier(hidden)
        if self.slot_classifier is not None:
            slot = self.slot_classifier(hidden)
        return OutputData(intent, slot)

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
        loss = 0
        intent_loss = None
        slot_loss = None
        if self.intent_classifier is not None:
            intent_loss = self.intent_classifier.compute_loss(pred, target) if compute_intent_loss else None
            intent_weight =  self.intent_classifier.config.get("weight")
            intent_weight = intent_weight if intent_weight is not None else 1.
            loss += intent_loss * intent_weight
        if self.slot_classifier is not None:
            slot_loss = self.slot_classifier.compute_loss(pred, target) if compute_slot_loss else None
            slot_weight = self.slot_classifier.config.get("weight")
            slot_weight = slot_weight if slot_weight is not None else 1.
            loss += slot_loss * slot_weight
        return loss, intent_loss, slot_loss


class StackPropagationDecoder(BaseDecoder):

    def forward(self, hidden: HiddenData):
        # hidden = self.interaction(hidden)
        pred_intent = self.intent_classifier(hidden)
        # embedding = pred_intent.output_embedding if pred_intent.output_embedding is not None else pred_intent.classifier_output
        # hidden.update_intent_hidden_state(torch.cat([hidden.get_slot_hidden_state(), embedding], dim=-1))
        hidden = self.interaction(pred_intent, hidden)
        pred_slot = self.slot_classifier(hidden)
        return OutputData(pred_intent, pred_slot)

class DCANetDecoder(BaseDecoder):

    def forward(self, hidden: HiddenData):
        if self.interaction is not None:
            hidden = self.interaction(hidden, intent_emb=self.intent_classifier, slot_emb=self.slot_classifier)
        return OutputData(self.intent_classifier(hidden), self.slot_classifier(hidden))

