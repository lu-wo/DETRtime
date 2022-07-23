from config import config
import logging
import torch
import os
import numpy as np
import re



def collate_fn(batch, num_classes=3):
    """
    receives batch B * 1000 * C 1000 seq --> subsample to 500
    :param batch: ndarray
    """
    data = [item[0] for item in batch]
    target = np.array([item[1] for item in batch])
    print(target.shape)
    print(np.squeeze(target).shape)
    y_tensor = torch.from_numpy(np.squeeze(target))
    print(y_tensor.shape)
    target = torch.nn.functional.one_hot(torch.from_numpy(np.array(target)), num_classes=num_classes)
    return [data, target]


class Ensemble_torch:
    """
    The Ensemble is a model itself, which contains a number of models whose prediction is averaged (majority decision in case of a classifier). 
    """

    def __init__(self, path, model_name='CNN', nb_models=5, loss='dice-loss', batch_size=64, **model_params):
        """
        :param path: path to the ensemble of models
        :param model_name: name of the model
        :param nb_models: number of models in the ensemble
        :param loss: loss function
        :param batch_size: batch size
        :param model_params: parameters of the model
        ...
        """
        self.model_name = model_name
        self.nb_models = nb_models
        self.model_params = model_params
        self.batch_size = batch_size
        self.loss = loss
        self.path = path 
        self.model_instance = None
        self.load_file_pattern = re.compile(model_name[:3] +  '.*.nb.*pth', re.IGNORECASE)
        self.models = []

        if self.model_name == 'CNN':
            from DL_Models.torch_models.CNN.CNN import CNN
            self.model = CNN
        elif self.model_name == 'EEGNet':
            from DL_Models.torch_models.EEGNet.eegNet import EEGNet
            self.model = EEGNet
        elif self.model_name == 'InceptionTime':
            from DL_Models.torch_models.InceptionTime.InceptionTime import Inception
            self.model = Inception
        elif self.model_name == 'PyramidalCNN':
            from DL_Models.torch_models.PyramidalCNN.PyramidalCNN import PyramidalCNN
            self.model = PyramidalCNN
        elif self.model_name == 'Xception':
            from DL_Models.torch_models.Xception.Xception import XCEPTION
            self.model = XCEPTION
        elif self.model_name == 'UNet':
            from DL_Models.torch_models.UNet.UNet import UNet
            self.model = UNet
        elif self.model_name == 'LSTM':
            from DL_Models.torch_models.LSTM.LSTM import LSTM
            self.model = LSTM
        elif self.model_name == 'ConvLSTM':
            from DL_Models.torch_models.ConvLSTM.ConvLSTM import ConvLSTM
            self.model = ConvLSTM
        elif self.model_name == 'biLSTM':
            from DL_Models.torch_models.BiLSTM.biLSTM import biLSTM
            self.model = biLSTM
        elif self.model_name == "SalientSleepNet":
            from DL_Models.torch_models.SalientSleepNet.SalientSleepNet import U2
            self.model = U2

    def fit(self, train_dataloader, validation_dataloader):
        """
        Fit an ensemble of models. They will be saved by BaseNet into the model dir
        :param train_dataloader: train dataloader
        :param validation_dataloader: validation dataloader
        """
        # Fit the models 
        for i in range(self.nb_models):
            logging.info(":---------------------------------------")
            logging.info('Start fitting model number {}/{} ...'.format(i+1, self.nb_models))
            
            if config['pretrained']:
                model = self.model(model_name=self.model_name, path=self.path, loss=self.loss, model_number=0, batch_size=self.batch_size, **self.model_params)  
                pretrained_path = './pretrained_models/pytorch/' + config['dataset'] + '/' + self.model_name + '_' + config['task'] + '.pth'
                model.load_state_dict(torch.load(pretrained_path))  
                model.eval()
            else:
                model = self.model(model_name=self.model_name, path=self.path, loss = self.loss, model_number=i, batch_size=self.batch_size, **self.model_params)
            logging.info(f"Fitting {self.model_name}")

            model.fit(train_dataloader, validation_dataloader)
            self.models.append(model)
            logging.info(":---------------------------------------")
            logging.info('Finished fitting model number {}/{} ...'.format(i+1, self.nb_models))


    def predict(self, test_dataloader):
        """
        Predict the test set using the ensemble of models
        :param test_dataloader: test dataloader
        """
        logging.info(f"Predicting with an ensemble of {len(self.models)} model(s).")
        for i, model in enumerate(self.models):
            logging.info(f"Predicting with: {self.model_name}")
            model.predict(test_dataloader)
            logging.info(f"FInished test set prediction.")
            

    def save(self, path):
        """
        Save the ensemble of models
        :param path: path to save the ensemble
        """
        for i, model in enumerate(self.models):
            ckpt_dir = path + self.model_name + '_nb_{}'.format(i) + '.pth'
            torch.save(model.state_dict(), ckpt_dir)

    def load(self, path):
        """
        Load the ensemble of models from path 
        :param path: path to the ensemble of models
        """
        self.models = []
        for file in os.listdir(path):
            if not self.load_file_pattern.match(file):
                continue
            logging.info(f"Loading model nb from file {file} and predict with it")
            model = self.model(model_name=self.model_name, path=path, loss=self.loss, model_number=0, batch_size=self.batch_size,
                               **self.model_params)  
            model.load_state_dict(torch.load(path + file)) 
            model.eval()  
            self.models.append(model)