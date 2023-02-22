'''
Author: Qiguang Chen
Date: 2023-01-11 10:39:26
LastEditors: Qiguang Chen
LastEditTime: 2023-01-26 17:18:22
Description: Root Model Module

'''
from torch import nn

from common.utils import OutputData, InputData
from model.decoder.base_decoder import BaseDecoder
from model.encoder.base_encoder import BaseEncoder


class OpenSLUModel(nn.Module):
    def __init__(self, encoder: BaseEncoder, decoder:BaseDecoder, **config):
        """Create model automatedly

        Args:
            encoder (BaseEncoder): encoder created by config
            decoder (BaseDecoder): decoder created by config
            config (dict): any other args
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

    def forward(self, inp: InputData) -> OutputData:
        """ model forward

        Args:
            inp (InputData): input ids and other information

        Returns:
            OutputData: pred logits
        """
        return self.decoder(self.encoder(inp))

    def decode(self, output: OutputData, target: InputData=None):
        """ decode output

        Args:
            pred (OutputData): pred logits data
            target (InputData): golden data

        Returns: decoded ids
        """
        return self.decoder.decode(output, target)

    def compute_loss(self, pred: OutputData, target: InputData, compute_intent_loss=True, compute_slot_loss=True):
        """ compute loss

        Args:
            pred (OutputData): pred logits data
            target (InputData): golden data
            compute_intent_loss (bool, optional): whether to compute intent loss. Defaults to True.
            compute_slot_loss (bool, optional): whether to compute slot loss. Defaults to True.

        Returns: loss value
        """
        return self.decoder.compute_loss(pred, target, compute_intent_loss=compute_intent_loss,
                                         compute_slot_loss=compute_slot_loss)
