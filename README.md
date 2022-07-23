## DETRtime: A Deep Learning Approach for the Segmentation of Electroencephalography Data in Eye Tracking Applications

DETRtime is a novel framework for time-series segmentation that creates ocular event detectors that does not require additionally recorded eye-tracking modality and rely solely on EEG data. Our end-to-end deep learning-based framework brings recent advances in Computer Vision to the forefront of the times series segmentation of EEG data.
## Overview

The repository consists of general functionality to run the DETRtime model and it's supplementing Baseline architectures on all benchmarked datasets. The Baselines contain state-of-the-art architectures such as U-Net, SalientSleepNet, InceptionTime, Xception, and many more. 

All models can be run on different segmentation datasets. We provide 4 ocular event datasets (large grid paradigm and visual symbol search as well as the real-world paradigms of watching movies and reading text). In addition to that, the models can be benchmarked on the publicly available Sleep-EDF-153 dataset for sleep stage segmentation available here: [sleep-edf](https://www.physionet.org/content/sleep-edfx/1.0.0/).

## Installation (Environment)

There are many dependencies for our code and we propose to use anaconda as package manager.
### General Requirements

Create a new conda environment:

```bash
conda create -n eegeyenet_benchmark python=3.8.5 
```

First install the general_requirements.txt

```bash
conda install --file general_requirements.txt 
```

### Pytorch Requirements

If you want to run the pytorch DL models, first install pytorch in the recommended way. For Linux users with GPU support this is:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch 
```

For other installation types and cuda versions, visit [pytorch.org](https://pytorch.org/get-started/locally/).

## DETRtime configuration
Everything related to DETRtime can be found in the DETRtime directory. 
#### Hyperparameter configuration 
Hyperparameters for DETRtime are read via the command line. We provide a run and evaluation shell script to save the current run configuration. For a detailed list of possible hyperparameter adaptions check the argument parser in DETRtime/main.py
#### How to run the model 
To run DETRtime, either execute 
```
python DETRtime/main.py --args
```
or make use of our shell script and run 
```
./DETRtime/train_model.sh
```

Evaluation can be done by setting the --eval flag, we provide the following script to run evaluation:
```
./DETRtime/eval_model.sh
```
## Baseline configuration 
#### Hyperparameter configuration 
In hyperparameters.py we define our baseline models. Models are configured in a dictionary which contains the object of the model and hyperparameters that are passed when the object is instantiated.

You can add your own models in the your_models dictionary. Specify the models for each task separately. Make sure to enable all the models that you want to run in config.py.

#### How to run the models 
Training related settings can be found in config.py: 

We start by explaining the settings that can be made: 

```bash
config['dataset'] = 'dots'
```

Include your own models as specified in hyperparameters.py. For instructions on how to create your own custom models see further below.

```bash
config['include_your_models'] = True
```

You can either choose to train models or use existing ones in /run/ and perform inference with them. Set

```bash
config['retrain'] = True 
config['save_models'] = True 
```

to train your specified models. Set both to False if you want to load existing models and perform inference. 
In this case specify the path to your existing model directory under

```bash
config['load_experiment_dir'] = path/to/your/model 
```

config.py further also allows to configure hyperparameters such as the learning rate, and enable early stopping of models.

To start the baseline benchmark, run

```bash
python3 main.py
```

A directory of the current run is created, containing a training log, saving console output and model checkpoints of all runs.


## Add Custom Models

For custom models we use a common interface we call trainer. A trainer is an object that implements the following methods:

```bash
fit() 
predict() 
save() 
load() 
```

#### Implementation of custom models

To implement your own custom model make sure that you create a class that implements the above methods. If you use library models, make sure to wrap them into a class that implements above interface used in our benchmark.

#### Adding custom models to our benchmark pipeline

In hyperparameters.py add your custom models into the your_models dictionary. You can add objects that implement the above interface. Make sure to enable your custom models in config.py.

