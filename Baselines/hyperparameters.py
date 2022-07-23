from DL_Models import Ensemble
from DL_Models.Ensemble import Ensemble
from config import config

# Use the following formats to add your own models (see hyperparameters.py for examples)
# 'NAME' : [MODEL, {'param1' : value1, 'param2' : value2, ...}]
# the model should be callable with MODEL(param1=value1, param2=value2, ...)
your_models = {
     'Segmentation_task' : {
            'movie' : {      
                'min' : {

                }
            }
     }
}

"""
Additional input and training parameters for the models
"""
electrodes = 3 if config['dataset'] == 'sleep' else 129 
timesamples = 60000 if config['dataset'] == 'sleep' else 500
input_shape = (timesamples, electrodes)
nb_models = 1
batch_size = 32
depth = 5
epochs = 1
verbose = True
seg_loss = 'dice-loss' # dice weights are set in config.py based on the dataset

"""
Hyperparameters of our deep learning baseline model suite are defined here.
"""
our_DL_models = {

    'Segmentation_task' : {

        'dots' : {
            'min' : {

                'Xception' : [Ensemble, {'model_name': 'Xception', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                          'nb_outlayer_channels': 3, 'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 128, 'verbose' : verbose, 'use_residual' : True, 'depth' : 12}],
                'CNN' : [Ensemble, {'model_name': 'CNN', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                         'nb_outlayer_channels': 3,'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 64, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                'PyramidalCNN' : [Ensemble, {'model_name': 'PyramidalCNN', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                          'nb_outlayer_channels': 3, 'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : False, 'depth' : depth}],
                'InceptionTime' : [Ensemble, {'model_name': 'InceptionTime', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                          'nb_outlayer_channels': 3, 'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 32, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                'EEGNet' : [Ensemble, {'model_name' : 'EEGNet', 'nb_models' : nb_models, 'loss':seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                          'nb_outlayer_channels': 3, 'epochs' : epochs, 'F1' : 32, 'F2' : 256, 'verbose' : verbose, 'D' : 4, 'kernel_size' : 16, 'dropout_rate' : 0.5}],
                'UNet' : [Ensemble, {'model_name': 'UNet', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                  'nb_outlayer_channels': 3, 'epochs' : epochs, 'verbose' : verbose}],
                'ConvLSTM' : [Ensemble, {'model_name': 'ConvLSTM', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                              'nb_outlayer_channels': 3,'kernel_size': 32, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth, 'hidden_size': 64, 'dropout':0.5}],
                'LSTM' : [Ensemble, {'model_name': 'LSTM', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                    'nb_outlayer_channels': 3, 'kernel_size': 32, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth, 'hidden_size': 64, 'dropout':0.5}],
                'biLSTM' : [Ensemble, {'model_name': 'biLSTM', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                        'nb_outlayer_channels': 3, 'kernel_size': 32, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth, 'hidden_size': 64, 'dropout':0.5}],
                'SalientSleepNet' : [Ensemble, {'model_name': 'SalientSleepNet', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                   'nb_outlayer_channels': 3, 'epochs' : epochs, 'verbose' : verbose}],
                
                }
        },
        'movie' : {
            'min' : {
                
                'CNN' : [Ensemble, {'model_name': 'CNN', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                        'nb_outlayer_channels': 3,'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 64, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                'PyramidalCNN' : [Ensemble, {'model_name': 'PyramidalCNN', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                         'nb_outlayer_channels': 3, 'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : False, 'depth' : depth}],
                'LSTM' : [Ensemble, {'model_name': 'LSTM', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                   'nb_outlayer_channels': 3, 'kernel_size': 8, 'epochs' : epochs, 'nb_filters' : 64, 'verbose' : verbose, 'use_residual' : True, 'depth' : 10, 'hidden_size': 256, 'dropout':0.5}],
                'biLSTM' : [Ensemble, {'model_name': 'biLSTM', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                       'nb_outlayer_channels': 3, 'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth, 'hidden_size': 128, 'dropout':0.5}],
                'ConvLSTM' : [Ensemble, {'model_name': 'ConvLSTM', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                             'nb_outlayer_channels': 3,'kernel_size': 32, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth, 'hidden_size': 64, 'dropout':0.5}],
                'SalientSleepNet' : [Ensemble, {'model_name': 'SalientSleepNet', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                   'nb_outlayer_channels': 3, 'epochs' : epochs, 'verbose' : verbose}],
                'UNet' : [Ensemble, {'model_name': 'UNet', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                   'nb_outlayer_channels': 3, 'epochs' : epochs, 'verbose' : verbose}],
                'EEGNet' : [Ensemble, {'model_name' : 'EEGNet', 'nb_models' : nb_models, 'loss':seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                         'nb_outlayer_channels': 3, 'epochs' : epochs, 'F1' : 32, 'F2' : 256, 'verbose' : verbose, 'D' : 4, 'kernel_size' : 16, 'dropout_rate' : 0.5}],
                'Xception' : [Ensemble, {'model_name': 'Xception', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                         'nb_outlayer_channels': 3, 'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 128, 'verbose' : verbose, 'use_residual' : True, 'depth' : 12}],
                'InceptionTime' : [Ensemble, {'model_name': 'InceptionTime', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                         'nb_outlayer_channels': 3, 'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 32, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                
                }
        },
        'vss' : {
            'min' : {

                'EEGNet' : [Ensemble, {'model_name' : 'EEGNet', 'nb_models' : nb_models, 'loss':seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                         'nb_outlayer_channels': 3, 'epochs' : epochs, 'F1' : 32, 'F2' : 256, 'verbose' : verbose, 'D' : 4, 'kernel_size' : 16, 'dropout_rate' : 0.5}],
                'Xception' : [Ensemble, {'model_name': 'Xception', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                         'nb_outlayer_channels': 3, 'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 128, 'verbose' : verbose, 'use_residual' : True, 'depth' : 12}],
                'InceptionTime' : [Ensemble, {'model_name': 'InceptionTime', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                         'nb_outlayer_channels': 3, 'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 32, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                'CNN' : [Ensemble, {'model_name': 'CNN', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                        'nb_outlayer_channels': 3,'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 64, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                'PyramidalCNN' : [Ensemble, {'model_name': 'PyramidalCNN', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                         'nb_outlayer_channels': 3, 'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : False, 'depth' : depth}],
                'UNet' : [Ensemble, {'model_name': 'UNet', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                   'nb_outlayer_channels': 3, 'epochs' : epochs, 'verbose' : verbose}],
                'SalientSleepNet' : [Ensemble, {'model_name': 'SalientSleepNet', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                   'nb_outlayer_channels': 3, 'epochs' : epochs, 'verbose' : verbose}],
                'ConvLSTM' : [Ensemble, {'model_name': 'ConvLSTM', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                             'nb_outlayer_channels': 3,'kernel_size': 32, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth, 'hidden_size': 64, 'dropout':0.5}],
                'LSTM' : [Ensemble, {'model_name': 'LSTM', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                   'nb_outlayer_channels': 3, 'kernel_size': 32, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth, 'hidden_size': 64, 'dropout':0.5}],
                'biLSTM' : [Ensemble, {'model_name': 'biLSTM', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                       'nb_outlayer_channels': 3, 'kernel_size': 32, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth, 'hidden_size': 64, 'dropout':0.5}]
                
                }
        },
        'zuco' : {
            'min' : {
                
                'CNN' : [Ensemble, {'model_name': 'CNN', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                        'nb_outlayer_channels': 3,'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 64, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                'SalientSleepNet' : [Ensemble, {'model_name': 'SalientSleepNet', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                  'nb_outlayer_channels': 3, 'epochs' : epochs, 'verbose' : verbose}],
                'ConvLSTM' : [Ensemble, {'model_name': 'ConvLSTM', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                             'nb_outlayer_channels': 3,'kernel_size': 32, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth, 'hidden_size': 64, 'dropout':0.5}],
                'LSTM' : [Ensemble, {'model_name': 'LSTM', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                   'nb_outlayer_channels': 3, 'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth, 'hidden_size': 128, 'dropout':0.5}],
                'biLSTM' : [Ensemble, {'model_name': 'biLSTM', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                       'nb_outlayer_channels': 3, 'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth, 'hidden_size': 128, 'dropout':0.5}],
                'UNet' : [Ensemble, {'model_name': 'UNet', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                   'nb_outlayer_channels': 3, 'epochs' : epochs, 'verbose' : verbose}],
                'EEGNet' : [Ensemble, {'model_name' : 'EEGNet', 'nb_models' : nb_models, 'loss':seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                         'nb_outlayer_channels': 3, 'epochs' : epochs, 'F1' : 32, 'F2' : 256, 'verbose' : verbose, 'D' : 4, 'kernel_size' : 16, 'dropout_rate' : 0.5}],
                'Xception' : [Ensemble, {'model_name': 'Xception', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                         'nb_outlayer_channels': 3, 'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 128, 'verbose' : verbose, 'use_residual' : True, 'depth' : 12}],
                'InceptionTime' : [Ensemble, {'model_name': 'InceptionTime', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                         'nb_outlayer_channels': 3, 'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 32, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                'PyramidalCNN' : [Ensemble, {'model_name': 'PyramidalCNN', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                         'nb_outlayer_channels': 3, 'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : False, 'depth' : depth}],

                }
        },
        'sleep' : {
            'min' : {

                'PyramidalCNN' : [Ensemble, {'model_name': 'PyramidalCNN', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                         'nb_outlayer_channels': 5, 'kernel_size': 32, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : False, 'depth' : depth}],
                'Xception' : [Ensemble, {'model_name': 'Xception', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                         'nb_outlayer_channels': 5, 'kernel_size': 32, 'epochs' : epochs, 'nb_filters' : 128, 'verbose' : verbose, 'use_residual' : True, 'depth' : 12}],
                'InceptionTime' : [Ensemble, {'model_name': 'InceptionTime', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                         'nb_outlayer_channels': 5, 'kernel_size': 32, 'epochs' : epochs, 'nb_filters' : 32, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                'CNN' : [Ensemble, {'model_name': 'CNN', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                        'nb_outlayer_channels': 5,'kernel_size': 32, 'epochs' : epochs, 'nb_filters' : 64, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                'EEGNet' : [Ensemble, {'model_name' : 'EEGNet', 'nb_models' : nb_models, 'loss':seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                         'nb_outlayer_channels': 5, 'epochs' : epochs, 'F1' : 32, 'F2' : 256, 'verbose' : verbose, 'D' : 4, 'kernel_size' : 16, 'dropout_rate' : 0.5}],
                'SalientSleepNet' : [Ensemble, {'model_name': 'SalientSleepNet', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                   'nb_outlayer_channels': 5, 'epochs' : epochs, 'verbose' : verbose}],
                'UNet' : [Ensemble, {'model_name': 'UNet', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                   'nb_outlayer_channels': 5, 'epochs' : epochs, 'verbose' : verbose}],
                'ConvLSTM' : [Ensemble, {'model_name': 'ConvLSTM', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                             'nb_outlayer_channels': 5,'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth, 'hidden_size': 64, 'dropout':0.5}],
                 'LSTM' : [Ensemble, {'model_name': 'LSTM', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                   'nb_outlayer_channels': 5, 'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth, 'hidden_size': 64, 'dropout':0.5}],
                'biLSTM' : [Ensemble, {'model_name': 'biLSTM', 'nb_models' : nb_models, 'loss': seg_loss, 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : config['window_len'],
                                       'nb_outlayer_channels': 5, 'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth, 'hidden_size': 64, 'dropout':0.5}],
            
            }
        }
    }
}



# merge two dict, new_dict overrides base_dict in case of incompatibility
def merge_models(base_dict, new_dict):
    result = dict()
    keys = base_dict.keys() | new_dict.keys()
    for k in keys:
        if k in base_dict and k in new_dict:
            if type(base_dict[k]) == dict and type(new_dict[k]) == dict:
                result[k] = merge_models(base_dict[k], new_dict[k])
            else:
                # overriding
                result[k] = new_dict[k]
        elif k in base_dict:
            result[k] = base_dict[k]
        else:
            result[k] = new_dict[k]
    return result


all_models = dict()

if config['include_DL_models']:
    all_models = merge_models(all_models, our_DL_models)
if config['include_your_models']:
    all_models = merge_models(all_models, your_models)
