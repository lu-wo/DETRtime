import torch.nn as nn
from .ConvNet import ConvNet
from .modules import TCSConv1d
import torch


class XCEPTION(ConvNet):
    """
    The Xception architecture. This is inspired by Xception paper, which describes how 'extreme' convolutions can be represented
    as separable convolutions and can achieve better accuracy than the Inception architecture. It is made of modules in a specific depth.
    Each module, in our implementation, consists of a separable convolution followed by batch normalization and a ReLu activation layer.
    """

    def __init__(self, input_shape, output_shape, kernel_size=40, nb_filters=64, use_residual=True, depth=6, maxpools = []):
        """
        nb_features: specifies number of channels before the output layer 
        """
        self.nb_features = nb_filters
        super(XCEPTION, self).__init__(input_shape=input_shape, output_shape=output_shape, kernel_size=kernel_size,
                                       nb_filters=nb_filters, use_residual=use_residual, depth=depth, maxpools = maxpools)

    def _module(self, depth):
        """
        The module of Xception. Consists of a separable convolution followed by batch normalization and a ReLu activation function.
        Padding=same 
        """
        return nn.Sequential(
            TCSConv1d(mother=self, depth=depth, bias=False),
            nn.BatchNorm1d(num_features=self.nb_features),
            nn.ReLU()
        )


if __name__ == '__main__':
    bs, chan, length = 16, 129, 500
    x = torch.randn(bs, length, chan)
    model = XCEPTION(input_shape=(129, 500), output_shape=None, kernel_size=64, nb_filters=16,
                     use_residual=False, depth=12)
    print(model(x).shape)
