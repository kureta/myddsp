import functools
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from myddsp.constants import CREPE_N_BINS

###########################################################################
# Model definition
###########################################################################


def layer(
    x: Tensor,
    conv: nn.Conv2d,
    batch_norm: nn.BatchNorm2d,
    padding: Tuple[int, int, int, int] = (0, 0, 31, 32),
) -> Tensor:
    """Forward pass through one layer."""
    x = F.pad(x, padding)
    x = conv(x)
    x = F.relu(x)
    x = batch_norm(x)
    return F.max_pool2d(x, (2, 1), (2, 1))


class Crepe(nn.Module):
    """Crepe model definition."""

    def __init__(self, model: str = "full"):
        super().__init__()

        # Model-specific layer parameters
        if model == "full":
            in_channels = [1, 1024, 128, 128, 128, 256]
            out_channels = [1024, 128, 128, 128, 256, 512]
            self.in_features = 2048
        elif model == "tiny":
            in_channels = [1, 128, 16, 16, 16, 32]
            out_channels = [128, 16, 16, 16, 32, 64]
            self.in_features = 256
        else:
            raise ValueError(f"Model {model} is not supported")

        # Shared layer parameters
        kernel_sizes = [(512, 1)] + 5 * [(64, 1)]
        strides = [(4, 1)] + 5 * [(1, 1)]

        # Overload with eps and momentum conversion given by MMdnn
        batch_norm_fn = functools.partial(nn.BatchNorm2d, eps=0.0010000000474974513, momentum=0.0)

        # Layer definitions
        self.conv1 = nn.Conv2d(
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
        )
        self.conv1_BN = batch_norm_fn(num_features=out_channels[0])

        self.conv2 = nn.Conv2d(
            in_channels=in_channels[1],
            out_channels=out_channels[1],
            kernel_size=kernel_sizes[1],
            stride=strides[1],
        )
        self.conv2_BN = batch_norm_fn(num_features=out_channels[1])

        self.conv3 = nn.Conv2d(
            in_channels=in_channels[2],
            out_channels=out_channels[2],
            kernel_size=kernel_sizes[2],
            stride=strides[2],
        )
        self.conv3_BN = batch_norm_fn(num_features=out_channels[2])

        self.conv4 = nn.Conv2d(
            in_channels=in_channels[3],
            out_channels=out_channels[3],
            kernel_size=kernel_sizes[3],
            stride=strides[3],
        )
        self.conv4_BN = batch_norm_fn(num_features=out_channels[3])

        self.conv5 = nn.Conv2d(
            in_channels=in_channels[4],
            out_channels=out_channels[4],
            kernel_size=kernel_sizes[4],
            stride=strides[4],
        )
        self.conv5_BN = batch_norm_fn(num_features=out_channels[4])

        self.conv6 = nn.Conv2d(
            in_channels=in_channels[5],
            out_channels=out_channels[5],
            kernel_size=kernel_sizes[5],
            stride=strides[5],
        )
        self.conv6_BN = batch_norm_fn(num_features=out_channels[5])

        self.classifier = nn.Linear(in_features=self.in_features, out_features=CREPE_N_BINS)

    def forward(self, x: Tensor, embed: bool = False) -> Tensor:
        # Forward pass through first five layers
        x = self.embed(x)

        if embed:
            return x

        # Forward pass through layer six
        x = layer(x, self.conv6, self.conv6_BN)

        # shape=(batch, self.in_features)
        x = x.permute(0, 2, 1, 3).reshape(-1, self.in_features)

        # Compute activation
        return torch.sigmoid(self.classifier(x))

    ###########################################################################
    # Forward pass utilities
    ###########################################################################

    def embed(self, x: Tensor) -> Tensor:
        """Map input audio to pitch embedding."""
        # shape=(batch, 1, 1024, 1)
        x = x[:, None, :, None]

        # Forward pass through first five layers
        x = layer(x, self.conv1, self.conv1_BN, (0, 0, 254, 254))
        x = layer(x, self.conv2, self.conv2_BN)
        x = layer(x, self.conv3, self.conv3_BN)
        x = layer(x, self.conv4, self.conv4_BN)
        x = layer(x, self.conv5, self.conv5_BN)

        return x


def load_model(capacity: str = "full") -> Crepe:
    """Preloads model from disk."""
    # Bind model and capacity
    model = Crepe(capacity)

    # Load weights
    this_dir = Path(__file__).parent.parent.parent
    file = this_dir / f"checkpoints/crepe/{capacity}.pth"
    model.load_state_dict(torch.load(file))

    return model
