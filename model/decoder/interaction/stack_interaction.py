import os
import torch
from torch import nn

from common import utils
from common.utils import ClassifierOutputData, HiddenData
from model.decoder.interaction.base_interaction import BaseInteraction


class StackInteraction(BaseInteraction):
    def __init__(self, **config):
        super().__init__(**config)
        self.intent_embedding = nn.Embedding(
            self.config["intent_label_num"], self.config["intent_label_num"]
        )
        self.differentiable = config.get("differentiable")
        self.intent_embedding.weight.data = torch.eye(
            self.config["intent_label_num"])
        self.intent_embedding.weight.requires_grad = False

    def forward(self, intent_output: ClassifierOutputData, encode_hidden: HiddenData):
        if not self.differentiable:
            _, idx_intent = intent_output.classifier_output.topk(1, dim=-1)
            feed_intent = self.intent_embedding(idx_intent.squeeze(2))
        else:
            feed_intent = intent_output.classifier_output
        encode_hidden.update_slot_hidden_state(
            torch.cat([encode_hidden.get_slot_hidden_state(), feed_intent], dim=-1))
        return encode_hidden

    @staticmethod
    def from_configured(configure_name_or_file="stack-interaction", **input_config):
        return utils.from_configured(configure_name_or_file,
                                     model_class=StackInteraction,
                                     config_prefix="./config/decoder/interaction",
                                     **input_config)
