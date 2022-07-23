# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import logging
import time

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

import util.misc as utils
from util.sequence_generator import generate_sequence_targets, generate_sequence_predictions



def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, timestamps: int, max_norm: float = 0):
    """
    train loop for DETRtime
    :param model: nn.Module
    :param criterion: nn.Module, Hunagiarn matcher
    :param data_loader: Iterable
    :param optimizer: torch.optimizer training
    :param device: device to operate on
    :param epoch: int denoting epoch
    :param timestamps: sequence length
    :param max_norm: upperbound used for gradient clipping
    :return:
    """
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    log_count = 0
    y_true = []
    y_hat = []

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_value = losses.item()
        if log_count % print_freq == 0:
            logging.info(loss_dict)
            logging.info({'loss': losses})

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            logging.info("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            logging.info(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict)
        metric_logger.update(class_error=loss_dict['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        log_count += 1

        for batch_idx in range(len(samples)):
            seq_true = generate_sequence_targets(targets[batch_idx], timestamps)

            d = {
                'pred_boxes': outputs['pred_boxes'][batch_idx].detach().cpu(),
                'pred_logits': outputs['pred_logits'][batch_idx].detach().cpu()
            }
            seq_hat = generate_sequence_predictions(d, timestamps)

            y_true.append(seq_true)
            y_hat.append(seq_hat)

    logging.info(f"Averaged stats: \n{metric_logger}")
    # log and print classification report
    y_true = np.array(y_true).flatten()
    y_hat = np.array(y_hat).flatten()
    logging.info(f"TRAIN CLASSIFICATION REPORT: \n {classification_report(y_true, y_hat, digits=4)}")
    cm = confusion_matrix(y_true, y_hat)
    logging.info(f"Confusion matrix training: \n{cm}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, epoch, device, output_dir, timestamps: int):
    """
    validation loop
    :param model: model to be trained
    :param criterion: Hungarian matcher
    :param data_loader: iterable over dataset
    :param epoch: number denoting epoch
    :param device: device to operate on
    :param output_dir: directory to log information
    :param timestamps: sequence length
    :return:
    """
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 100

    log_count = 0
    y_true = []
    y_hat = []

    acc_time = 0
    num_batches = 0 
    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        start = time.time()
        outputs = model(samples)
        total_time = time.time() - start 
        acc_time += total_time
        num_batches += 1 
        if num_batches > 500:
            break 

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_value = losses.item()

        if log_count % print_freq == 0:
            logging.info(loss_dict)
            logging.info({'loss': losses})

        metric_logger.update(loss=loss_value, **loss_dict)
        metric_logger.update(class_error=loss_dict['class_error'])
        log_count += 1

        # compute sequences for classification report
        for batch_idx in range(len(samples)):
            seq_true = generate_sequence_targets(targets[batch_idx], timestamps)
            d = {
                'pred_boxes': outputs['pred_boxes'][batch_idx].detach().cpu(),
                'pred_logits': outputs['pred_logits'][batch_idx].detach().cpu()
            }
            seq_hat = generate_sequence_predictions(d, timestamps)

            y_true.append(seq_true)
            y_hat.append(seq_hat)

    # gather the stats from all processes
    logging.info(f"Averaged stats:")
    logging.info(metric_logger)

    # log and print classification report
    y_true = np.array(y_true).flatten()
    y_hat = np.array(y_hat).flatten()
    logging.info(f"VALID CLASSIFICATION REPORT: \n {classification_report(y_true, y_hat, digits=4)}")

    print('Confusion Matrix Valid:')
    cm = confusion_matrix(y_true, y_hat)
    # cm = cm/cm.astype(np.float).sum(axis=1)
    logging.info(f"Confusion matrix: \n{cm}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
