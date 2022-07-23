"""
    Create a custom Pyramidal CNN backbone
    It only implements the Pyramidal CNN module that is then stacked in a CNN in ConvNet
"""
import torch
from torch import nn
from .ConvNet import ConvNet
from .modules import Pad_Conv, Pad_Pool


class PyramidalCNN(ConvNet):
    """
    The Classifier_PyramidalCNN is one of the simplest convolutional architectures. It implements the class ConvNet, which is made of modules with a
    specific depth, where for each depth the number of filters is increased.
    """

    def __init__(self, input_shape, output_shape, kernel_size=16, nb_filters=16, use_residual=False, depth=6, maxpools = maxpools):
        """
        nb_features: specifies number of channels before the output layer 
        """
        self.nb_features = depth * nb_filters  # For pyramidal we increase the nbfilters each depth layer
        super().__init__(input_shape=input_shape, output_shape=output_shape, kernel_size=kernel_size,
                         nb_filters=nb_filters, use_residual=use_residual, depth=depth, maxpools = [])

    def _module(self, depth):
        """
        The module of CNN is made of a simple convolution with batch normalization and ReLu activation. Finally, MaxPooling is also used.
        The number of filters / output channels is increases with the depth of the model.
        Padding=same 
        """
        return nn.Sequential(
            Pad_Conv(kernel_size=self.kernel_size),
            nn.Conv1d(in_channels=self.nb_channels if depth == 0 else depth * self.nb_filters,
                      out_channels=(depth + 1) * self.nb_filters,
                      kernel_size=self.kernel_size,
                      bias=False),
            nn.BatchNorm1d(num_features=(depth + 1) * self.nb_filters),
            nn.ReLU(),
            Pad_Pool(left=0, right=1),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )


if __name__ == '__main__':
    bs, chan, length = 16, 129, 500
    x = torch.randn(bs, length, chan)
    model = PyramidalCNN(input_shape=(129, 500), output_shape=None, kernel_size=64, nb_filters=16,
                         use_residual=False, depth=12)
    print(model(x).shape)
