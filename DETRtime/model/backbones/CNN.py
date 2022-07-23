"""
    Create a custom Pyramidal CNN backbone
    It only implements the Pyramidal CNN module that is then stacked in a CNN in ConvNet
"""
import torch
from torch import nn
from typing import Dict
from .ConvNet import ConvNet
from .modules import Pad_Conv, Pad_Pool


class CNN(ConvNet):
    """
    The CNN is one of the simplest classifiers. It implements the class ConvNet, which is made of modules with a specific depth.
    """

    def __init__(self, input_shape, output_shape, kernel_size=64, nb_filters=16, use_residual=True, depth=12, maxpools = []):
        """
        nb_features: specifies number of channels before the output layer 
        """
        self.nb_features = nb_filters  # For CNN simply the number of filters / channels
        super().__init__(input_shape=input_shape, output_shape=output_shape, kernel_size=kernel_size,
                         nb_filters=nb_filters, use_residual=use_residual, depth=depth, maxpools = maxpools)

    def _module(self, depth):
        """
        The module of CNN is made of a simple convolution with batch normalization and ReLu activation. Finally, MaxPooling is used.
        We use two custom padding modules such that keras-like padding='same' is achieved, i.e. tensor shape stays constant when passed through the module.
        """
        return nn.Sequential(
            Pad_Conv(kernel_size=self.kernel_size, value=0),
            nn.Conv1d(in_channels=self.nb_channels if depth == 0 else self.nb_features,
                      out_channels=self.nb_features, kernel_size=self.kernel_size, bias=False),
            nn.BatchNorm1d(num_features=self.nb_features),
            nn.ReLU(),
            Pad_Pool(left=0, right=1, value=0),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )


if __name__ == '__main__':
    bs, chan, length = 16, 129, 500
    x = torch.randn(bs, length, chan)
    model = CNN(input_shape=(129, 500), output_shape=None, kernel_size=64, nb_filters=16,
                         use_residual=True, depth=12)
    print(model(x).shape)