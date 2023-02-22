from torch import nn

class BaseInteraction(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config

    def forward(self, hidden1, hidden2):
        NotImplementedError("no implemented")
