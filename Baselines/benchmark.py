import numpy as np
import logging
import time
from config import config
from hyperparameters import all_models
import os

from DL_Models.torch_models.torch_utils.dataset import create_dataloader, create_biased_dataloader



def try_models(models, N=1):
    """
    Here we evaluate all models defined in hyperparameters.py for the given task and dataset.
    """

    logging.info("Training the models")
    for name, model in models.items():
        logging.info("Training of " + name)

        for i in range(N):
            # create the model with the corresponding parameters
            path = config['checkpoint_dir'] + 'run' + str(i + 1) + '/'
            trainer = model[0](path, **model[1])

            # Create dataloaders
            logging.info(f"Loading train and val data.")
            data = np.load(config['data_dir'] + '/tensors/train.npz')
            train_dataloader = create_biased_dataloader(data['EEG'], data['labels'], model[1]['batch_size'])
            data = np.load(config['data_dir'] + '/tensors/val.npz')
            validation_dataloader = create_dataloader(data['EEG'], data['labels'], model[1]['batch_size'])
            logging.info(f"Loaded train and val data.")

            start_time = time.time()

            if not os.path.exists(path):
                os.makedirs(path)

            if config['retrain']:
                trainer.fit(train_dataloader, validation_dataloader)
            else:
                trainer.load(path)

            logging.info(f"Start loading test data.")
            test_data = np.load(config['data_dir'] + '/tensors/test.npz')
            test_dataloader = create_dataloader(test_data['EEG'], test_data['labels'], model[1]['batch_size'])
            logging.info(f"Loaded test data.")
            trainer.predict(test_dataloader)
            runtime = (time.time() - start_time)

            logging.info("--- Runtime: %s for seconds ---" % runtime)
            logging.info('------------------------------------------------------------------------------------')
            logging.info('------------------------------------------------------------------------------------')
        

def benchmark():
    """
    Entry point to the benchmark experiment 
    """
    np.savetxt(config['model_dir']+'/config.csv', [config['task'], config['dataset'], config['preprocessing']], fmt='%s')
    models = all_models[config['task']][config['dataset']][config['preprocessing']]
    try_models(models, N=1)
