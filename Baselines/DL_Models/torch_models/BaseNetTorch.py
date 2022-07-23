import torch
import sys
from torch import nn
from config import config
import logging
from DL_Models.torch_models.torch_utils.training import train_loop, validation_loop
import wandb
import logging
from config import config
import torch
import wandb 
from DL_Models.torch_models.Modules import Pad_Pool
import time 


class BaseNet(nn.Module):
    """
    BaseNet class to inherit common functionality shared by all models 
    """

    def __init__(self, model_name, path, loss, input_shape, output_shape, epochs=50, verbose=True, model_number=0):
        """
        Initialize common variables of models based on BaseNet, e.g. ConvNet or EEGNET 
        Create the common output layer dependent on the task to run 
        """
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape 
        self.epochs = epochs
        self.verbose = verbose
        self.model_number = model_number
        self.timesamples = self.input_shape[0]
        self.nb_channels = self.input_shape[1]
        self.early_stopped = False
        self.loss = loss
        self.path = path
        self.model_name = model_name

        # Create output layer depending on the loss function 
        if loss == 'dice-loss':
            from DL_Models.torch_models.torch_utils.DiceLoss import DiceLoss
            dice_weights = config['dice_weights']
            logging.info(f"Dice weights: {dice_weights}")
            self.loss_fn = DiceLoss(
                weight=torch.tensor(dice_weights),
                p=2,
                reduction='mean')
            self.output_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.get_nb_channels_output_layer(), 
                    out_channels=min(self.get_nb_features_output_layer(), self.nb_outlayer_channels),
                    kernel_size=1,
                    stride=1
                ),
                nn.BatchNorm1d(num_features=self.nb_outlayer_channels),
                nn.ReLU(),
                Pad_Pool(left=0, right=1, value=0),
                nn.MaxPool1d(kernel_size=2, stride=1)
            )
            self.out_linear = nn.Sequential(
                nn.Linear(500, self.output_shape), 
                #nn.Softmax() # this is done in Dice Loss 
            )
        else:
            raise ValueError("Choose a valid loss function")

        logging.info(f"Using loss fct: {self.loss_fn}")

    # abstract method
    def forward(self, x):
        """
        Implements a forward pass of the network 
        This method has to be implemented by models based on BaseNet 
        """
        pass

    # abstract method
    def get_nb_features_output_layer(self):
        """
        Return the number of features that the output layer should take as input
        This method has to be implemented by models based on BaseNet to compute the number of hidden neurons that the output layer takes as input. 
        """
        pass

    # abstract method
    def get_nb_channels_output_layer(self):
        """
        Return the number of channels that the convolution before output layer should take as input to reduce them to 1 channel
        This method has to be implemented by models based on BaseNet to compute the number of hidden neurons that the output layer takes as input. 
        """
        pass

    # abstract method
    def _split_model(self):
        pass

    def fit(self, train_dataloader, validation_dataloader):
        """
        Fit the model on the dataset defined by data x and labels y 
        :param train_dataloader: Dataloader for the training set
        :param validation_dataloader: Dataloader for the validation set
        """
        # Move the model to GPU
        if torch.cuda.is_available():
            self.cuda()
            logging.info(f"Model moved to cuda")
        # Create the optimizer
        optimizer = torch.optim.AdamW(
            list(self.parameters()), lr=config['learning_rate'], weight_decay=1e-4)
        # Create datastructures to collect metrics and implement early stopping
        epochs = self.epochs
        metrics = {'train_loss': [], 'val_loss': [], 'train_f1_macro': [], 'val_f1_macro': []}
        best_val_loss = sys.maxsize  # For early stopping
        patience = 0

        # Train the model
        for t in range(epochs):
            logging.info("-------------------------------")
            logging.info(f"Epoch {t+1}")
            # Run through training and validation set
            if not self.early_stopped:
                start_time = time.time()
                train_loss_epoch, train_acc_epoch, train_f1_weighted = train_loop(
                    train_dataloader, self.float(), self.loss_fn, optimizer)
                logging.info("--- Training %s seconds ---" % (time.time() - start_time))
                start_time = time.time()
                logging.info("Validating")
                val_loss_epoch, val_acc_epoch, val_f1_weighted = validation_loop(
                    validation_dataloader, self.float(), self.loss_fn)
                logging.info("--- Validation %s seconds ---" % (time.time() - start_time))
                metrics['train_loss'].append(train_loss_epoch)
                metrics['val_loss'].append(val_loss_epoch)
                metrics['train_f1_macro'].append(train_f1_weighted)
                metrics['val_f1_macro'].append(val_f1_weighted)
            else:
                break  # early stopped
            # Impementation of early stopping and model checkpoint
            if config['early_stopping'] and not self.early_stopped:
                if patience > config['patience']:
                    logging.info(f"Early stopping the model after {t} epochs")
                    self.early_stopped = True
                if val_loss_epoch >= best_val_loss:
                    logging.info(
                        f"Validation loss did not improve, best was {best_val_loss}")
                    patience += 1
                else:
                    best_val_loss = val_loss_epoch
                    logging.info(
                        f"Improved validation loss to: {best_val_loss}")
                    self.save()  # save the new best model
                    patience = 0

    def predict(self, test_dataloader):
        """
        Predict the labels of the test set
        """
        logging.info(f"---- PREDICTING ON TEST SET ")
        validation_loop(test_dataloader, self.float(), self.loss_fn)

    def save(self):
        """
        Save the model to the path built below 
        """
        ckpt_dir = self.path + self.model_name + \
            '_nb_{}'.format(self.model_number) + '.pth'
        torch.save(self.state_dict(), ckpt_dir)
        logging.info(f"Saved new best model to ckpt_dir")
