import argparse
import datetime
import random
import time
from pathlib import Path
import logging
import sys 

import numpy as np
import torch

from dataset.dataset import create_dataloader
from engine import evaluate, train_one_epoch
from model import build_model
from util.misc import collate_fn



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # Hyperparameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--wandb_dir', default='noname', type=str)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=50, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    # parser.add_argument('--frozen_weights', type=str, default=None,
    #                    help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='pcnn', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--kernel_size', default=64, type=int)
    parser.add_argument('--nb_filters', default=16, type=int)
    parser.add_argument('--in_channels', default=129, type=int)
    parser.add_argument('--out_channels', default=1, type=int)
    parser.add_argument('--backbone_depth', default=6, type=int)
    parser.add_argument('--use_residual', default=False, type=bool)
    # parser.add_argument('--dilation', action='store_true',
    #                    help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--back_channels', default=16, type=int,
                        help='Defines the number of embedding channels')
    parser.add_argument('--back_layers', default=12, type=int,
                        help='Defines the number of layers in the backbone')
    parser.add_argument('--maxpools', nargs='+', type=int, help="Optionally define maxpools for ConvNet classes")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=20, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    # parser.add_argument('--masks', action='store_true',
    #                    help="Train segmentation head if the flag is provided")

    # Loss
    # parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
    #                    help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=10, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    # parser.add_argument('--mask_loss_coef', default=1, type=float)
    # parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=10, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.2, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--sleep', default=False, type=bool, help='sleep annotations')
    parser.add_argument('--num_classes', default=3, type=int,
                        help="number of classes to predict")
    parser.add_argument('--timestamps', default=500, type=int)
    parser.add_argument('--timestamps_output', default=500, type=int)
    # parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--scaler', type=str, choices=['Standard', 'MaxMin'])
    # parser.add_argument('--coco_panoptic_path', type=str)
    # parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='', required=True,
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--apply_labels', default=False, type=bool)
    # distributed training parameters
    # parser.add_argument('--world_size', default=1, type=int,
    #                    help='number of distributed processes')
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):

    # Initialize logger file used everywhere throughout the run
    logging.basicConfig(filename=args.output_dir + '/' + f'{time.time()}info.log', level=logging.INFO)
    logging.info('Started the Logging')
    logging.info(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of model params: {n_parameters}')

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    # define optimizer
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # create scaler
    scaler = None
    do_scaling = False
    if args.scaler:
        do_scaling = True
        if args.scaler == 'MaxMin':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif args.scaler == 'Standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        else:
            raise NotImplementedError(f'Scaler {args.scaler} not supported.')

    if not args.eval:
        # Create dataloaders
        logging.info("Loading training data.")
        logging.info("Using train_30")
        scaler, data_loader_train = create_dataloader(
            data_dir=args.data_path,
            file=f'train_no_neg1.npz',
            validation=False,
            batch_size=args.batch_size,
            workers=args.num_workers,
            collate_fn=collate_fn,
            standard_scale=do_scaling,
            scaler=scaler,
            max_queries=args.num_queries,
            num_classes=args.num_classes,
            sleep = args.sleep,
            apply_label_dict = args.apply_labels
        )

        logging.info("Loading validation data.")
        _, data_loader_val = create_dataloader(
                                data_dir=args.data_path, 
                                file=f'val_no_neg1.npz',
                                validation=True,
                                batch_size=args.batch_size,
                                workers=args.num_workers,
                                collate_fn=collate_fn,
                                standard_scale=do_scaling,
                                scaler = scaler,
                                max_queries = args.num_queries,
                                num_classes = args.num_classes,
                                sleep = args.sleep,
                                apply_label_dict = args.apply_labels
                                )

    output_dir = Path(args.output_dir)

    if args.resume:
        logging.info(f"Resume training with model.")

        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        logging.info(f"Loaded model.")

        old_args = checkpoint['args']

        assert old_args.scaler == args.scaler, "scaler should be equal"

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        # dataset_test = build_dataset(image_set='test', args=args)
        # sampler_test = torch.utils.data.RandomSampler(dataset_test)
        # data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
        #                       drop_last=False, collate_fn=None, num_workers=args.num_workers)
        logging.info(f"Evaluation mode.")
        logging.info(f"Loading test data.")
        _, data_loader_test = create_dataloader(
            data_dir=args.data_path,
            file=f'test_no_neg1.npz',
            validation=True,
            batch_size=args.batch_size,
            workers=args.num_workers,
            collate_fn=collate_fn,
            standard_scale=do_scaling,
            scaler=scaler,
            sleep = args.sleep,
            apply_label_dict = args.apply_labels
        )

        logging.info("Start Evaluation")
        test_stats = evaluate(
            model, criterion, data_loader_test, 0, device, args.output_dir, args.timestamps
        )

        logging.info(f"Test stats:")
        logging.info(test_stats)
        logging.info(f"Finished logging.")
        return

    logging.info(f"Start training.")
    logging.info(f"-------------------------------------------------------------------------------")
    start_time = time.time()
    # For saving best model 
    min_val_loss = sys.maxsize
    for epoch in range(args.start_epoch, args.epochs):
        logging.info(f"--------------- Epoch {epoch} --------------------")
        # Train one epoch
        logging.info(f"Training...")
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.timestamps_output, args.clip_max_norm
        )
        # HARDCODED LR DROP !
        for i in range(20):
            lr_scheduler.step()

        # save model every epoch
        if args.output_dir:
            checkpoint_paths = [] #[output_dir / f'checkpoint_{epoch}.pth']
            # extra checkpoint before LR drop and every 100 epochs
            #if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
            #save every epochs
            checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

            for checkpoint_path in checkpoint_paths:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        logging.info(f"Validation...")
        # run eval
        test_stats = evaluate(
            model, criterion, data_loader_val, epoch, device, args.output_dir, args.timestamps_output
        )

        if test_stats['loss'] < min_val_loss:
            logging.info(f"New minimum validation loss. Saving as checkpoint_best_val.pth !")
            min_val_loss = test_stats['loss']
            checkpoint_path = output_dir / f'checkpoint_best_val.pth'
            torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path) 


        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        logging.info(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info('Training time {}'.format(total_time_str))
    logging.info(f"Finished training.")
    logging.info(f"-------------------------------------------------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
