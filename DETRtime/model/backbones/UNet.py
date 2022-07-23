import logging
import math
import torch
import torch.nn as nn
import torchvision
import logging

class Block(nn.Module):

    def __init__(self, in_filters, out_filters, kernel_size=5, stride=1, padding=2, activation=nn.ReLU()):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_filters, out_filters, kernel_size, stride=1, padding=2,
                      dilation=1, groups=1, bias=True,
                      padding_mode='zeros'),
            activation,
            nn.BatchNorm1d(out_filters),
            nn.Conv1d(out_filters, out_filters, kernel_size, stride=1, padding=2,
                      dilation=1, groups=1, bias=True,
                      padding_mode='zeros'),
            activation,
            nn.BatchNorm1d(out_filters)
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):

    def __init__(self, output_size=1024, kernel_size=5, filters=(129, 128,  256, 512), pools=(4, 3, 2)):
        super(Encoder, self).__init__()
        assert ((len(filters) - 1) == len(pools))
        self.encBlocks = nn.ModuleList([Block(filters[i], filters[i + 1], kernel_size)
                                        for i in range(len(filters) - 1)])
        self.pools = [nn.MaxPool1d(pools[i]) for i in range(len(pools))]
        self.output = Block(filters[-1], output_size)

    def forward(self, x):
        output = []
        for i, block in enumerate(self.encBlocks):
            x = block(x)
            #print(f'Encoder Layer {i} shape:{x.shape}')
            output.append(x)
            x = self.pools[i](x)
            #print(f'Encoder MaxPool {i} shape:{x.shape}')
        x = self.output(x)
        #print(f'Final Encoder Layer shape:{x.shape}')
        output.append(x)
        return output


class Decoder(nn.Module):
    """
        expects input of size (B, T, C), returns (B, T, K)
    """

    def __init__(self, filters=(1024, 512, 256, 128), kernel_size=5, pools=(2, 2, 3)):
        super(Decoder, self).__init__()
        self.upconv = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=pools[i], mode='linear'),
                nn.Conv1d(filters[i], filters[i + 1], kernel_size=pools[i], padding=1)
            )
            for i in range(len(pools))])
        self.decBlocks = nn.ModuleList([Block(filters[i], filters[i + 1], kernel_size)
                                        for i in range(len(filters) - 1)])

    def forward(self, x, encblocks):
        # loop through the number of channels
        #channel numbers don't match up
        for i in range(len(self.decBlocks)):
            # pass the inputs through the upsampler blocks
            x = self.upconv[i](
                x)  # should reshape exactly such that after concatenation we again have doubling of everything
            #print(f'Upsample Layer {i} shape:{x.shape}')
            encFeat = self.crop(encblocks[i], x)
            x = torch.cat([x, encFeat], dim=1)
            #print(f'Crop Layer {i} shape:{x.shape}')
            x = self.decBlocks[i](x)
            #print(f'Decoder Layer {i} shape:{x.shape}')
        # return the final decoder output
        return x

    def crop(self, encFeatures, x):
        _, c, t = x.size()
        encFeatures = torchvision.transforms.CenterCrop([c, t])(encFeatures)
        return encFeatures


class UNet(nn.Module):

    def __init__(self, input_shape = (500, 3), output_shape = (0, 0), kernel_size = 5, nb_filters = 64,
                 use_residual = False, depth = 4,
                 encChannels=(128, 256, 512), bottleneck=1024,
                 decChannels=(512, 256, 128), pools=(3, 2, 2), maxpools = [], 
                 retainDim=True):
        """

        :param input_shape:
        :param encChannels:
        :param bottleneck:
        :param decChannels:
        :param pools: used for determining down-/upsampling
        :param kernel_size:
        :param nb_filters:
        :param retainDim:
        :param use_crf:
        :param maxpools: Spurious Argument for backbone
        :param output_shape: unused dummy for backbone
        :param use_residual: unused dummy for backbone
        :param depth: unuse dummy for backbone
        """
        #encChannels=(128, 256, 512), bottleneck=1024,
        #         decChannels=(512, 256, 128), pools=(3,2,2)
        logging.info("Building UNet")
        logging.info(f' Encoder channels {encChannels}')
        logging.info(f' Decoder channels {decChannels}')
        logging.info(f' bottleneck {bottleneck}')
        logging.info(f' Pools {pools}')
        super(UNet, self).__init__()
        self.input_shape = input_shape
        self.seq_len = self.input_shape[0]
        input_channel = self.input_shape[1]
        kernel_size = 5
        
        self.encChannels = (input_channel,) + encChannels
        self.decChannels = (bottleneck,) + decChannels
        self.encoder = Encoder(filters=self.encChannels, output_size=bottleneck, pools=pools, kernel_size=kernel_size)
        self.decoder = Decoder(filters=self.decChannels, pools=pools[::-1], kernel_size=kernel_size)

        self.output = nn.Sequential(
            nn.Conv1d(self.decChannels[-1], nb_filters,  1),
            nn.Sigmoid()
        )

        self.nb_features = nb_filters #dummy
        #Dice Loss
        # self.loss = DiceLoss(
        #     weight=torch.tensor([1.1, 5.2, 10.1]),
        #     p=2,
        #     reduction='mean')
        #
    def forward(self, x):
        """

        :param x: expects x of shape (bs, c, sq)
        :return:
        """
        #x = x.permute(0, 2, 1)
        batchsize, channels,seq_length = x.size()
        encFeatures = self.encoder(x)
        decFeatures = self.decoder(encFeatures[::-1][0],
                                   encFeatures[::-1][1:])
        segment_map = self.output(decFeatures)
        _, _,  segment_length = segment_map.size()
        # pad to fit
        top = math.floor((seq_length - segment_length) / 2)
        bottom = math.ceil(( seq_length - segment_length) / 2)
        output = nn.ZeroPad2d(padding=(bottom, top, 0, 0))(segment_map)

        #output = output.permute(0, 2, 1)

        return output