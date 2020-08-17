from collections import OrderedDict

import torch.nn as nn

import math

DEFAULT_LAYER_CONFIG = [
    [383, 1024],
    [1024, 1024],
    [1024, 801]
]


def calculate_bias_bound(weights):
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in)
    return bound


def get_tanh_linear_layer(input_dim, output_dim):
    layer = nn.Linear(input_dim, output_dim)
    nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))

    bound = calculate_bias_bound(layer.weight)
    nn.init.uniform_(layer.bias, -bound, bound)
    return layer


def get_relu_linear_layer(input_dim, output_dim):
    layer = nn.Linear(input_dim, output_dim)
    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

    bound = calculate_bias_bound(layer.weight)
    nn.init.uniform_(layer.bias, -bound, bound)
    return layer


class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, layer_config, non_linearity="relu"):
        super(DNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_config = layer_config
        self.non_linearity = non_linearity
        self.model = self.compose_model()

    def forward(self, batch):
        return self.model(batch)

    def compose_model(self, layer_config=None):
        if layer_config is None:
            layer_config = self.layer_config
        layers = OrderedDict()
        for idx, config in enumerate(layer_config):
            input_dim, output_dim = config
            if self.non_linearity == "tanh":
                layer = get_tanh_linear_layer(input_dim, output_dim)
                layers['linear-%d' % idx] = layer
                if idx != len(layer_config) - 1:
                    layers['nonlinear-%d' % idx] = nn.Tanh()
            else:
                layer = get_relu_linear_layer(input_dim, output_dim)
                layers['linear-%d' % idx] = layer
                if idx != len(layer_config) - 1:
                    layers['nonlinear-%d' % idx] = nn.ReLU()

        return nn.Sequential(layers)
