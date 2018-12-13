import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_shape, hidden_size, num_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_action))

    def forward(self, x):
        return self.net(x)
