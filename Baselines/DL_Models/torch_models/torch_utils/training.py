import torch
import numpy as np
import logging
import torch
import time 
from sklearn.metrics import classification_report, confusion_matrix
from DL_Models.torch_models.torch_utils.DiceLoss import make_one_hot
from config import config 

TARGET_NAMES = ['Sleep stage W', 
                'Sleep stage 1',
        'Sleep stage 2',
        'Sleep stage 3/4', 
       'Sleep stage R'] if config['dataset'] == 'sleep' \
                else ['fixation', 'saccade', 'blink'] if config['dataset'] == 'sleep' else \
                ['fixation', 'saccade', 'blink']


def train_loop(dataloader, model, loss_fn, optimizer):
    """
    Performs one epoch of training the model through the dataset stored in dataloader, predicting one batch at a time
    Using the given loss_fn and optimizer
    Returns training metrics (loss for regressions, loss and accuracy for classification) of the epoch
    This function is called by BaseNet each epoch
    """
    size = len(dataloader)
    # num_datapoints = len(dataloader.dataset)
    training_loss, correct = 0, 0
    #
    model.train()
    predictions, correct_pred = [], []
    for batch, (X, y) in enumerate(dataloader):
        # Move tensors to GPU
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
        pred = model(X)
        y = y.squeeze()
        bs, sq = y.size()
        y_dice = y.long().reshape(-1, 1)
        y_dice = make_one_hot(y_dice, 5) if config['dataset'] == 'sleep' else make_one_hot(y_dice, 3)
        if torch.cuda.is_available():
            y_dice = y_dice.cuda()
        pred_dice = pred.reshape(bs * sq, -1)
        loss = loss_fn(pred_dice, y_dice)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Add up metrics
        training_loss += loss.item()
        _, pred_indices = torch.max(pred_dice, 1)
        predictions += list(pred_indices.cpu().numpy().flat)
        correct_indices = y.long().flatten()
        correct_pred += list(correct_indices.cpu().numpy())
        # Remove batch from gpu
        del X
        del y
        torch.cuda.empty_cache()

    loss = training_loss / size
    logging.info(f"Avg training loss: {loss:>7f}")
    return float(loss)


def validation_loop(dataloader, model, loss_fn):
    """
    Performs one epoch of training the model through the dataset stored in dataloader, predicting one batch at a time
    Using the given loss_fn and optimizer
    Returns training metrics (loss for regressions, loss and accuracy for classification) of the epoch
    This function is called by BaseNet each epoch
    """
    model.eval()
    size = len(dataloader)
    validation_loss, correct = 0, 0
    # Init lists for overall metrics
    predictions, ground_truth = [], []
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # Move tensors to GPU
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            pred = model(X)
            y = y.long()
            y = y.squeeze()
            bs, sq = y.size()
            y_dice = y.long().reshape(-1, 1)
            y_dice = make_one_hot(y_dice, 5) if config['dataset'] == 'sleep' else make_one_hot(y_dice, 3)
            y_dice = y_dice.reshape(bs * sq, -1)
            if torch.cuda.is_available():
                y_dice = y_dice.cuda()
            pred_dice = pred.reshape(bs * sq, -1)
            loss = loss_fn(pred_dice, y_dice)

            _, pred_indices = torch.max(pred_dice, 1)
            validation_loss += loss.item()
            predictions += list(pred_indices.cpu().numpy().flat)
            ground_truth += list(y.long().flatten().cpu().numpy())
            del X
            del y
            torch.cuda.empty_cache()

    logging.info("Calculating validation metrics")
    # Compute validation metrics
    loss = validation_loss / size
    logging.info(f"Avg validation loss: {loss:>7f}")
    val_scores = classification_report(ground_truth, predictions, target_names=TARGET_NAMES, digits=3)
    logging.info(f"Validation performance: \n"
                 f" {val_scores}")
    return float(loss), 0, 0


def test_loop(dataloader, model):
    predictions, correct_pred = [], []
    with torch.no_grad():
        model.eval()
        for batch, (X, y) in enumerate(dataloader):
            # Move tensors to GPU
            if torch.cuda.is_available():
                X = X.cuda()
            pred = model(X)
            pred_dice = pred.reshape(-1, 3)
            _, pred_indices = torch.max(pred_dice, 1)
            predictions += list(pred_indices.cpu().numpy().flat)
            correct_indices = y.long().flatten()
            correct_pred += list(correct_indices.cpu().numpy())
            del X
            del y
            torch.cuda.empty_cache()
    train_scores = classification_report(correct_pred, predictions, target_names=TARGET_NAMES, digits=3)
    logging.info(f"Test performance: \n"
                 f"{train_scores}")
    logging.info(f"Test Confusion matrix: \n{confusion_matrix(correct_pred, predictions)}")
    return np.array(correct_pred), np.array(predictions)
