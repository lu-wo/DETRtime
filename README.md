## DETRtime: A Deep Learning Approach for the Segmentation of Electroencephalography Data in Eye Tracking Applications
#### Published in the [proceedings](https://proceedings.mlr.press/v162/wolf22a.html) of the 39th International Conference on Machine Learning (ICML) 2022
#### The paper can be found on [arXiv](https://arxiv.org/abs/2206.08672)

DETRtime is a novel framework for time-series segmentation. We use it to create an ocular event detector that does not require additionally recorded eye-tracking modality and rely solely on EEG data. Our end-to-end deep learning-based framework brings recent advances in Computer Vision to the forefront of the times series segmentation of EEG data. Compared to other time series segmentation solutions, we tackle the problem via instance segmentation (instead of semantic segmentation).

## Overview

The repository consists of general functionality to run the DETRtime model and it's supplementing Baseline architectures from the paper on all benchmarked datasets. The Baselines contain state-of-the-art architectures such as U-Net, SalientSleepNet, InceptionTime, Xception, and many more. 

All models can be run on different segmentation datasets. We provide 4 ocular event datasets (large grid paradigm and visual symbol search as well as the real-world paradigms of watching movies and reading text). In addition to that, the models can be benchmarked on the publicly available Sleep-EDF-153 dataset for sleep stage segmentation available here: [sleep-edf](https://www.physionet.org/content/sleep-edfx/1.0.0/).

## Installation (Environment)

There are many dependencies for our code and we propose to use anaconda as package manager.
### Requirements

We recommed to use [Anaconda](https://www.anaconda.com/) to create a new python environment:

```bash
conda create -n detrtime python=3.8.5 
```

To install PyTorch, run:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch 
```

For other installation types and cuda versions, visit [pytorch.org](https://pytorch.org/get-started/locally/).

## DETRtime configuration
Everything related to DETRtime can be found in the DETRtime directory. 
#### Hyperparameter configuration 
Hyperparameters for DETRtime are read via the command line. We provide a run and evaluation shell script to save the current run configuration. For a detailed list of possible hyperparameter adaptions check the argument parser in DETRtime/main.py
#### How to run the DETRtime model 
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

