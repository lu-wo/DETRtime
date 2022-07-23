import torch.nn as nn
from DL_Models.torch_models.Modules import Pad_Pool, Pad_Conv
import torch
from DL_Models.torch_models.BaseNetTorch import BaseNet
from config import config 

class ConvLSTM(BaseNet):
	"""
	CNN + LSTM Model
	Extract features from the CNN and feed them to the LSTM 
	"""
	def __init__(self, model_name, path, loss, model_number, batch_size, input_shape, output_shape, kernel_size=64, epochs = 50, nb_filters=16, verbose=True,
				use_residual=True, depth=12, nb_outlayer_channels = 3, hidden_size=129, dropout=0.5):
		"""
		nb_features: specifies number of channels before the output layer 
		"""
		self.nb_features = nb_filters  # For CNN simply the number of filters
		self.hidden_size = self.nb_features # LSTM processes CNN output channels
		self.nb_outlayer_channels = nb_outlayer_channels
		super().__init__(model_name=model_name, path=path, loss=loss, input_shape=input_shape, output_shape=output_shape, 
							epochs=epochs, verbose=verbose, model_number=model_number)
		self.timesamples = 500 if config['dataset'] == 'sleep' else input_shape[0]
		self.nb_channels = input_shape[1]
		self.use_residual = use_residual
		self.depth = depth
		self.kernel_size = kernel_size
		self.nb_filters = nb_filters
		
		# CNN
		self.conv_blocks = nn.ModuleList([self._module(d) for d in range(self.depth)])
		if self.use_residual:
			self.shortcuts = nn.ModuleList([self._shortcut(d) for d in range(int(self.depth / 3))])
		self.gap_layer = nn.AvgPool1d(kernel_size=2, stride=1)
		self.gap_layer_pad = Pad_Pool(left=0, right=1, value=0)

		# Helper Conv with large kernel to strip down the input width from 60000 to 110 
		self.scale_temporal = nn.Sequential(
				nn.Conv1d(in_channels=3, out_channels=3, kernel_size=5100, stride=500),
				nn.BatchNorm1d(num_features=3),
				nn.ReLU(),
				Pad_Pool(left=2, right=2, value=0),
				nn.MaxPool1d(kernel_size=5, stride=1)
				)
		self.scale_temporal = nn.Linear(60000, 500)
	
		# LSTM 
		self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, dropout=dropout, 
							num_layers=3, bidirectional=True)


	def forward(self, x):
		"""
		Implements the forward pass of the network
		Modules defined in a class implementing ConvNet are stacked and shortcut connections are used if specified.
		"""
		x = x.permute(0, 2, 1)
		if config['dataset'] == 'sleep':
			x = self.scale_temporal(x) # scale temporal due to the complexity of 60k sequence length 
		input_res = x  # set for the residual shortcut connection

		# Stack the CNN modules and residual connection
		shortcut_cnt = 0
		for d in range(self.depth):
			x = self.conv_blocks[d](x)
			if self.use_residual and d % 3 == 2:
				res = self.shortcuts[shortcut_cnt](input_res)
				shortcut_cnt += 1
				x = torch.add(x, res)
				x = nn.functional.relu(x)
				input_res = x
		# LSTM PART 				
		x = self.gap_layer_pad(x)
		x = self.gap_layer(x)
		x = x.permute(2, 0, 1)
		output, (hn, cn) = self.lstm(x)
		output = output.permute(1,2,0)
		output = self.output_layer(output)
		output = self.out_linear(output)
		output = output.permute(0, 2, 1)
		#print(f'ConvLstm output shape {output.size()}')
		return output

	def _shortcut(self, depth):
		"""
		Implements a shortcut with a convolution and batch norm
		This is the same for all models implementing ConvNet, therefore defined here
		Padding before convolution for constant tensor shape, similar to tensorflow.keras padding=same
		"""
		return nn.Sequential(
			Pad_Conv(kernel_size=self.kernel_size, value=0),
			nn.Conv1d(in_channels=self.nb_channels if depth == 0 else self.nb_features,
					  out_channels=self.nb_features, kernel_size=self.kernel_size),
			nn.BatchNorm1d(num_features=self.nb_features)
		)

	def get_nb_channels_output_layer(self):
		return 2 * self.hidden_size

	def get_nb_features_output_layer(self):
		"""
		Return number of features passed into the output layer of the network
		nb.features has to be defined in a model implementing ConvNet
		"""
		return 2 * self.hidden_size * self.timesamples 

	def _module(self, depth):
		"""
		The module of CNN is made of a simple convolution with batch normalization and ReLu activation. Finally, MaxPooling is used.
		We use two custom padding modules such that keras-like padding='same' is achieved, i.e. tensor shape stays constant when passed through the module.
		"""
		return nn.Sequential(
				Pad_Conv(kernel_size=self.kernel_size, value=0),
				nn.Conv1d(in_channels=self.nb_channels if depth==0 else self.nb_features, 
							out_channels=self.nb_features, kernel_size=self.kernel_size, bias=False,
						),
				nn.BatchNorm1d(num_features=self.nb_features),
				nn.ReLU(),
				Pad_Pool(left=2, right=2, value=0),
				nn.MaxPool1d(kernel_size=5, stride=1)
				)
