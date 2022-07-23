"""
Common interface to fit and predict with torch and tf ensembles 
"""
from config import config
import numpy as np
from DL_Models.torch_models.Ensemble_torch import Ensemble_torch


class Ensemble:

    def __init__(self, path, model_name, nb_models, loss, batch_size=64, **model_params):
        """
        This class provides an interface to create an ensemble of models, one can implement a tensorflow/keras version of Ensemble_torch and use the training pipeline 
        :param path: path to the folder where the models are stored
        :param model_name: name of the model to load
        :param nb_models: number of models to create as ensemble, if nb_models == 1 then we create a single model 
        :param loss: loss function to use as string 
        :param batch_size: batch size to use for training
        """
        self.path = path 
        self.ensemble = Ensemble_torch(path=path, model_name=model_name, nb_models=nb_models, loss=loss, batch_size=batch_size, **model_params)

    def fit(self):
        self.ensemble.fit()
        
    def predict(self):
        self.ensemble.predict()

    def save(self, path):
        self.ensemble.save(path)

    def load(self, path):
        self.ensemble.load(path)