from dataclasses import dataclass, field
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from src.utilities.convolutions import ConvParam, compute_flattened_size


@dataclass
class MNISTNetConfig:
    image_width: int = 28
    image_height: int = 28
    image_channels: int = 1
    conv_params: List[ConvParam] = field(default_factory=list)
    dropout: float = 0.25
    dropout2: float = 0.5
    fc1_out_features: int = 128
    fc2_out_features: int = 10


class MNISTNet(nn.Module):
    def __init__(self, config: MNISTNetConfig = MNISTNetConfig()):
        super(MNISTNet, self).__init__()
        self.config = config

        # Dynamically create convolutional layers based on conv_params
        self.convs = nn.ModuleList()
        for param in config.conv_params:
            self.convs.append(nn.Conv2d(param.in_channels, param.out_channels, param.kernel_size, param.stride,
                                        padding=param.padding))

        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout2)

        # Use compute_flattened_size to get the size after convolutions and optional pooling
        flattened_size = compute_flattened_size(config.conv_params, config.image_width, config.image_height)

        # Initialize fully connected layers using the computed flattened_size
        self.fc1 = nn.Linear(flattened_size, config.fc1_out_features)
        self.fc2 = nn.Linear(config.fc1_out_features, config.fc2_out_features)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = F.relu(x)
            # Apply pooling only if specified in the ConvParam
            if self.config.conv_params[i].pooling:
                x = F.max_pool2d(x, self.config.conv_params[i].pooling)
            x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
