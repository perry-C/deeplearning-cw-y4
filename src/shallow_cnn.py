from torch import nn, optim
from typing import NamedTuple

import torch


class InputShape(NamedTuple):
    height: int
    width: int
    channels: int


class SHA_CNN(nn.Module):
    def __init__(
            self,
            dropout_rate,
            height: int = 80,
            width: int = 80,
            channels: int = 1,
            # 10 music genres
            class_count: int = 10,
    ):
        super().__init__()

        self.input_shape = InputShape(
            height=height, width=width, channels=channels)
        self.class_count = class_count

        # ===========================================================================
        #                    Time architecture (left-pipeline)
        # ===========================================================================

        #  conv1 = 16 temporal filters of shape(10, 23)
        self.conv1_l = nn.Conv2d(
            in_channels=self.input_shape.channels,
            # out_channels = how many conv-filters at this layer
            out_channels=16,
            kernel_size=(10, 23),
            padding="same",
        )
        self.initialise_layer(self.conv1_l)
        # Add a layer for batch-normalization after convolution and before activation
        self.norm1_l = nn.BatchNorm2d(self.conv1_l.out_channels)
        # we apply Leaky ReLU activation with α=0.3
        self.actv1_l = nn.LeakyReLU(negative_slope=0.3)
        self.pool1_l = nn.MaxPool2d(kernel_size=(1, 20))

        # ===========================================================================
        #                    Frequency architecture (right-pipeline)
        # ===========================================================================

        #  conv1 = 16 temporal filters of shape(21, 20)
        self.conv1_r = nn.Conv2d(
            in_channels=self.input_shape.channels,
            # out_channels = how many conv-filters at this layer
            out_channels=16,
            kernel_size=(21, 20),
            padding="same",
        )
        self.initialise_layer(self.conv1_r)
        # Add a layer for batch-normalization after convolution and before activation
        self.norm1_r = nn.BatchNorm2d(self.conv1_r.out_channels)
        # we apply Leaky ReLU activation with α=0.3
        self.actv1_r = nn.LeakyReLU(negative_slope=0.3)
        self.pool1_r = nn.MaxPool2d(kernel_size=(20, 1))

        # ===========================================================================
        #                           Merged architecture
        # ===========================================================================

        # "... the shape of 1×10240, which serves as input to a 200 units
        # fully connected layer with a dropout of 10%."
        self.fc1_m = nn.Linear(10240, 200)
        self.initialise_layer(self.fc1_m)
        self.actv1_m = nn.LeakyReLU(negative_slope=0.3)

        self.drop1_m = nn.Dropout(dropout_rate)

        # "All networks were trained towards
        # categorical-crossentropy objective using the stochastic
        # Adam optimization [KB14] with beta1 =0.9, beta2 =0.999,
        # epsilon=1e−08 and a learning rate of 0.00005."

        # There is a missing layer within Figure 1 of the paper,
        # which you will need to map the 200 units to the number of classes for the output.
        self.fc2_m = nn.Linear(self.fc1_m.out_features, self.class_count)

        self.actv2_m = nn.Softmax(dim=1)

    def forward(self, audios: torch.Tensor) -> torch.Tensor:
        x_l = self.conv1_l(audios)
        x_l = self.norm1_l(x_l)
        x_l = self.actv1_l(x_l)
        x_l = self.pool1_l(x_l)

        x_r = self.conv1_r(audios)
        x_r = self.norm1_r(x_r)
        x_r = self.actv1_r(x_r)
        x_r = self.pool1_r(x_r)

        # The 16 feature maps of each pipeline are flattened
        # to a shape of 1×5120 and merged by concatenation into
        # the shape of 1×10240
        x_l = torch.flatten(x_l, start_dim=1)
        x_r = torch.flatten(x_r, start_dim=1)

        # ========================Layers after Merge============================
        x = torch.cat((x_l, x_r), 1)
        x = self.fc1_m(x)

        # Following typical architecture design, and for consistency with the deep
        # architecture/contents of the text, we also add a LeakyReLU with alpha=0.3 after the
        # 200 unit fully connected layer, before dropout, which is not shown in Figure 1.
        x = self.actv1_m(x)

        # the dropout is applied AFTER the 200 unit FULLY CONNECTED LAYER as they say in the text,
        # not before/after the merge as they show in the figure.
        x = self.drop1_m(x)
        x = self.fc2_m(x)

        x = self.actv2_m(x)

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
