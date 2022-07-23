# configuration used by the training and evaluation methods
# let's keep it here to have a clean code on other methods that we try
import time
import os
import numpy as np 
config = dict()

##################################################################
##################################################################
############### BENCHMARK CONFIGURATIONS #########################
##################################################################
##################################################################
config['task'] = 'Segmentation_task'
config['dataset'] = 'movie' # set to: 'dots', 'movie', 'vss', 'zuco', 'sleep'
config['preprocessing'] = 'min'  # we currently only provide minimial preprocessings
config['include_DL_models'] = True  # our baseline models, fixed to True 
config['include_your_models'] = False

config['biased_sampling'] = True # upsampling of minority class samples during training 
config['window_len'] = 500 
config['data_dir'] = "/itet-stor/wolflu/net_scratch/projects/ICML_Seg_Baselines/EEGEyeNet_experimental/data/segmentation/ICML_movie_min_segmentation_minseq_500_margin_1_amp_thresh_10000"
config['dice_weights'] = [0.866, 0.106, 0.027] # label distribution
config['dice_weights'] = np.array(config['dice_weights']) / np.array(config['dice_weights']).sum() # inverse the weights for the dice loss

##################################################################
##################################################################
############### PATH CONFIGURATIONS ##############################
##################################################################
##################################################################
# Where experiment results are stored.
config['log_dir'] = './runs/'
# Path to training, validation and test data folders.
#config['data_dir'] = './data/'
# Path of root
config['root_dir'] = '.'
# Retrain or load already trained
config['retrain'] = True
config['pretrained'] = False 
config['save_models'] = True
# If retrain is false we need to provide where to load the experiment files
config['load_experiment_dir'] = ''
# all_EEG_file should specify the name of the file where the prepared data is located (if emp
def build_file_name():
    all_EEG_file = config['task'] + '_with_' + config['dataset']
    all_EEG_file = all_EEG_file + '_' + 'synchronised_' + config['preprocessing']
    return all_EEG_file
#config['all_EEG_file'] = build_file_name() # or use your own specified file name
config['all_EEG_file'] = 'Segmentation_task_with_dots_synchronised_min.npz'
##################################################################
##################################################################
############### MODELS CONFIGURATIONS ############################
##################################################################
##################################################################
# Specific to models now
config['framework'] = 'pytorch' # pytorch or tensorflow 
config['learning_rate'] = 1e-3
config['early_stopping'] = True
config['patience'] = 10
config['biased_sampling'] = False 

##################################################################
############### HELPER VARIABLES #################################
##################################################################
##################################################################
config['trainX_variable'] = 'EEG'
config['trainY_variable'] = 'labels'


def create_folder():
    if config['retrain']:
        model_folder_name = str(int(time.time()))
        model_folder_name += '_' + config['dataset'] + '_' + config['preprocessing']

        config['model_dir'] = os.path.abspath(os.path.join(config['log_dir'], model_folder_name))
        config['checkpoint_dir'] = config['model_dir'] + '/checkpoint/'
        if not os.path.exists(config['model_dir']):
            os.makedirs(config['model_dir'])

        if not os.path.exists(config['checkpoint_dir']):
            os.makedirs(config['checkpoint_dir'])

        config['info_log'] = config['model_dir'] + '/' + 'info.log'
        config['batches_log'] = config['model_dir'] + '/' + 'batches.log'

    else:
        config['model_dir'] = config['log_dir'] + config['load_experiment_dir']
        config['checkpoint_dir'] = config['model_dir'] + 'checkpoint/'
        stamp = str(int(time.time()))
        config['info_log'] = config['model_dir'] + '/' + 'inference_info_' + stamp + '.log'
        config['batches_log'] = config['model_dir'] + '/' + 'inference_batches_' + stamp + '.log'
        