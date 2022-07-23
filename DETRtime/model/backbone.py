# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from torch import nn
from .position_encoding import build_position_encoding
#from backbones.PyramidalCNN import PyramidalCNN
from .position_encoding import Joiner
import logging


class Backbone(nn.Module):
    """
    Backbone class that contains the common functionality that a backbone provides
    """

    def __init__(self, input_shape, output_shape, model_class, kernel_size=32, nb_filters=16,
                 use_residual=False, depth=12, maxpools = []):
        super().__init__()
        logging.info("Building backbone")
        self.model = model_class(input_shape=input_shape, output_shape=output_shape, kernel_size=kernel_size,
                                 nb_filters=nb_filters, use_residual=use_residual, depth=depth, maxpools = maxpools)
        self.num_channels = self.model.nb_features

    def forward(self, x):
        return self.model(x)


def build_backbone(args):
    if args.backbone in ['cnn']:
        from .backbones.CNN import CNN
        model_class = CNN
        logging.info('[INFO] using CNN backbone')
    elif args.backbone in ['pcnn']:
        from .backbones.PyramidalCNN import PyramidalCNN
        model_class = PyramidalCNN
        logging.info('[INFO] Using PCNN backbone')
    elif args.backbone in ['xception']:
        from .backbones.Xception import XCEPTION
        model_class = XCEPTION
        logging.info('[INFO] Using Xception backbone')
    elif args.backbone in ['inception_time']:
        from .backbones.InceptionTime import Inception
        model_class = Inception
        logging.info('[INFO] Using InceptionTime backbone')
    elif args.backbone in ['unet']:
        from .backbones.UNet import UNet
        model_class = UNet
        logging.info('[INFO] Using UNet backbone')
    else:
        raise NotImplementedError("Choose a valid backbone model")

    logging.info("Building position encoding")
    # Build position encoding and backbone
    position_embedding = build_position_encoding(args)
    logging.info(f"Max Pool input {args.maxpools}")
    backbone = Backbone(
        input_shape=(args.timestamps, args.in_channels),
        output_shape=(args.timestamps, args.out_channels),
        model_class=model_class,
        kernel_size=args.kernel_size,
        nb_filters=args.nb_filters,
        use_residual=args.use_residual,
        depth=args.backbone_depth,
        maxpools = args.maxpools
                        )

    # Final model is joined backbone (CNN) output with position embedding
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    if args.backbone in ['pcnn']:
        model.num_channels = args.back_channels * args.back_layers

    return model
