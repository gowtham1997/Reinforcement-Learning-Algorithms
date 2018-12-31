import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math


class NoisyLinear(nn.Linear):

    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full(
            (out_features, in_features), sigma_init))
        self.register_buffer(
            "epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(
                torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features,))
        self.reset_parameters()

    def reset_parameters(self):
        # init weight and bias of linear layer as defined in the paper
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        if self.bias is not None:
            self.epsilon_bias.normal_()
            bias = self.bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input,
                        self.weight + self.sigma_weight * self.epsilon_weight,
                        bias)


class NoisyFactorizedLinear(nn.Linear):

    def __init__(self, in_features, out_features, sigma_zero=0.04, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.full((out_features,
                                                     in_features),
                                                    sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(
                torch.full((out_features,), sigma_init))

    def forward(self, input):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        def func(x):
            return torch.sign(x) * torch.sqrt(torch.abs(x))

        eps_in = func(self.epsilon_input)
        eps_out = func(self.epsilon_output)

        if self.bias is not None:
            bias = self.bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out)

        return F.linear(input, self.weight + noise_v * self.sigma_weight, bias)


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU())
        conv_out_size = self.get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def get_conv_out(self, input_shape):
        out = self.conv(torch.zeros(1, *input_shape)).shape
        return int(np.prod(out))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.shape[0], -1)
        return self.fc(conv_out)


class NoisyDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU())
        conv_out_size = self.get_conv_out(input_shape)
        self.noisy_layers = [
            NoisyLinear(conv_out_size, 512),
            NoisyLinear(512, num_actions)
        ]
        self.fc = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1]
        )

    def get_conv_out(self, input_shape):
        out = self.conv(torch.zeros(1, *input_shape)).shape
        return int(np.prod(out))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.shape[0], -1)
        return self.fc(conv_out)

    def noisy_layer_sigma_snr(self):
        return [
            ((layer.weight ** 2).mean().sqrt() /
             (layer.sigma_weight ** 2).mean().sqrt()).data.cpu().numpy()[0]
            for layer in self.noisy_layers
        ]
