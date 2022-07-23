import logging
from torch import nn
import torch
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv1d, Conv2d
from torch.nn.modules.pooling import MaxPool1d
from abc import ABC
from torch import nn
import torch
from abc import ABC
from DL_Models.torch_models.BaseNetTorch import BaseNet


class LSTM(BaseNet):

	def __init__(self, model_name, path, loss, model_number, batch_size, input_shape, output_shape, kernel_size=64, epochs = 50, nb_filters=16, verbose=True,
				use_residual=True, depth=10, nb_outlayer_channels = 3,hidden_size=129, dropout=0.5):
		"""
		We define the layers of the network in the __init__ function
		"""
		self.hidden_size = hidden_size
		self.timesamples = input_shape[0]
		self.input_channels = input_shape[1]
		self.nb_outlayer_channels = nb_outlayer_channels
		super().__init__(model_name=model_name, path=path, model_number=model_number, loss=loss, input_shape=input_shape, output_shape=output_shape, epochs=epochs, verbose=verbose)
		# Define the modules 
		self.lstm = nn.LSTM(input_size=self.input_channels, hidden_size=self.hidden_size, dropout=dropout, num_layers=depth)

	def forward(self, x, hidden=None):
		"""
		Implements the forward pass of the network
		Modules defined in a class implementing ConvNet are stacked and shortcut connections are used if specified. 
		"""
		x = x.permute(1,0,2)
		output, (hn, cn) = self.lstm(x, hidden)
		output = output.permute(1,2,0) 
		output = self.output_layer(output)
		output = self.out_linear(output)
		output = output.permute(0, 2, 1)
		return output

	def get_nb_channels_output_layer(self):
		return self.hidden_size

	def get_nb_features_output_layer(self):
		"""
		Return number of features passed into the output layer of the network 
		nb.features has to be defined in a model implementing ConvNet
		"""
		return self.hidden_size * self.timesamples 


if __name__ == "__main__":
	batch, chan, time = 16, 129, 500
	out_chan, out_width = 3, 500
	model = LSTM(input_shape=(time, chan), output_shape=(out_chan, out_width))
	tensor = torch.randn(batch, chan, time)
	out = model(tensor)
	print(f"output shape: {out.shape}")
