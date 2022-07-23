import torch
import torch.utils.data
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import logging
import os
from config import config 

class EEGSampler(torch.utils.data.sampler.Sampler):
    """
        samples with a bit higher weight sequences that either contain saccaes or blinks
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None):
        super(EEGSampler, self).__init__(dataset)

        self.dataset = dataset
        self.indices = list(range(dataset.current_length)) if indices is None else indices

    def __iter__(self):
        return (self.indices[i] for i in
                torch.multinomial(self.dataset.weights, self.dataset.current_length, replacement=True))

    def __len__(self):
        return len(self.dataset)


class EEGDataset(torch.utils.data.Dataset):
    """
        since dataloader length depends on dataset size, we have precalculate the average length
    """

    def __init__(self, data_dir, batch_size=32, seq_length=500, validation=False):
        super(EEGDataset, self).__init__()
        self.id = 0
        self.subjects = os.listdir(data_dir)
        self.data_dir = data_dir
        self.length = 0
        self.batch_size = batch_size
        self.seq_length = seq_length
        print(self.seq_length)
        for file in self.subjects:
            # print(file)
            data = np.load(self.data_dir + '/' + file)
            self.length += (len(data['labels']) // batch_size // seq_length)
        self.validation = validation
        self.load_new_user(self.id)

    def load_new_user(self, id):
        # Transform np.array to torch flaot tensor
        data = np.load(self.data_dir + '/' + self.subjects[id])
        X = data['EEG'].astype(np.float)
        conv = {'L_fixation': 0, 'L_saccade': 1, 'L_blink': 2,
                'R_fixation': 0,
                'R_saccade': 1,
                'R_blink': 2}
        func = np.vectorize(conv.get)
        y = func(data['labels']).astype(np.float)
        # reshaping samples
        length = len(y)
        cut = length // self.seq_length * self.seq_length
        X_cut = X[:cut, :129]
        y_cut = y[:cut]
        X_reshape = X_cut.reshape(-1, self.seq_length, 129)
        y_reshape = y_cut.reshape(-1, self.seq_length)
        tensor_x = torch.as_tensor(X_reshape).float()
        tensor_y = torch.as_tensor(y_reshape).float()
        # move models to cuda if available
        # if torch.cuda.is_available():
        #     tensor_x = tensor_x.cuda()
        #     tensor_y = tensor_y.cuda()
        self.dataset = TensorDataset(tensor_x, tensor_y)
        # new sampling weights recalculated for each user
        self.current_length = len(y_reshape)
        self.queries = 0
        # calculate weights

        arr = np.ones(self.current_length)
        patterns = [np.full(6, 1), np.full(6, 2)]  # saccade 1, blink 2
        indices = list(range(self.current_length))
        df = pd.DataFrame()
        # for i, pattern in enumerate(patterns):
        #     mask = np.apply_along_axis(lambda x: self.find_conv(x, pattern), -1, tensor_y)
        #     arr[mask] = i + 1
        only_fix_weight = 0.4
        blk_sac_weight = 1.0
        blk_weight = 1.0
        # print(f"only fix weight: {only_fix_weight}, blk/sacc weigth: {blk_sac_weight}")
        mask = np.apply_along_axis(lambda x: np.count_nonzero(x == 0) < 0.9 * self.seq_length, -1, tensor_y)
        arr[mask] = blk_sac_weight
        mask1 = np.apply_along_axis(lambda x: np.count_nonzero(x == 0) >= 0.9 * self.seq_length, -1, tensor_y)
        arr[mask1] = only_fix_weight
        # mask2 = np.apply_along_axis(lambda x: np.count_nonzero(x == 2) > 0, -1, tensor_y)
        # arr[mask2] = blk_weight
        df["label"] = arr
        df.index = indices
        df = df.sort_index()
        label_to_count = df["label"].value_counts()
        self.label_to_count = label_to_count

        # weights = 1.0/label_to_count[df["label"]]
        weights = label_to_count[df["label"]]
        if self.validation:
            arr = np.ones(self.current_length)
        self.weights = torch.DoubleTensor(arr.tolist())
        # self.weights = torch.DoubleTensor(weights.to_list())

    def find_conv(self, seq, subseq):
        target = np.dot(subseq, subseq)
        candidates = np.where(np.correlate(seq, subseq, mode='valid') == target)[0]
        check = candidates[:, np.newaxis] + np.arange(len(subseq))
        mask = np.all((np.take(seq, check) == subseq), axis=-1)
        return np.any(mask)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not self.validation:
            self.queries += 1
            if self.queries > (self.current_length // (self.batch_size * 2)):
                self.id += 1
                self.id = self.id % len(self.subjects)
                self.load_new_user(self.id)
                self.queries = 0
                return self.dataset[[i for i in torch.multinomial(self.weights, self.batch_size, replacement=True)]]
            else:
                return self.dataset[[i for i in torch.multinomial(self.weights, self.batch_size, replacement=True)]]
        else:
            self.queries += 1
            if self.queries >= (self.current_length // self.batch_size):
                print(f'Switching user at index {self.queries * self.batch_size}')
                print(f'maximum was {self.current_length // self.batch_size * self.batch_size}')
                self.id += 1
                self.id = self.id % len(self.subjects)
                self.load_new_user(self.id)
                self.queries = 0
                return self.dataset[self.queries * self.batch_size:(self.queries + 1) * self.batch_size]
            else:
                return self.dataset[self.queries * self.batch_size:(self.queries + 1) * self.batch_size]


def create_dataloader(X, y, batch_size, drop_last=True):
    """
    Input: X, y of type np.array
    Return: pytorch dataloader containing the dataset of X and y that returns batches of size batch_size
    """
    cat_dict = \
    {
        'Sleep stage 1': 1, 
        'Sleep stage 2': 2, 
        'Sleep stage 3': 3, 
        'Sleep stage 4': 3,
       'Sleep stage R': 4, 
       'Sleep stage W': 0
       } if config['dataset'] == 'sleep' else \
    {
        'L_fixation': 0,
        'L_saccade': 1,
        'L_blink': 2,
        'R_fixation': 0,
        'R_saccade': 1,
        'R_blink': 2
    } 
    # y[0:5, 1]
    if config['dataset'] != 'sleep':
        func = np.vectorize(cat_dict.get)
        y = func(y)
    #y=np.vstack(y).astype(np.float32)
    y = y.astype('float32') 
    # Transform np.array to torch flaot tensor
    tensor_x = torch.as_tensor(X).float()
    tensor_y = torch.as_tensor(y).float()

    if len(tensor_y.size()) > 2:
        tensor_y = tensor_y.squeeze()
        
    # Unsqueeze channel direction for eegNet model
    # if model_name == 'EEGNet':
    #     logging.info(f"Unsqueeze data for eegnet")
    #     tensor_x = tensor_x.unsqueeze(1)
    # Log the shapes
    logging.info(f"Tensor x size: {tensor_x.size()}")
    logging.info(f"Tensor y size: {tensor_y.size()}")
    # Set device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Create dataset and dataloader
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, num_workers=4)


def create_biased_dataloader(X, y, batch_size, drop_last=True, seq_length = 500):
    """
    Input: X, y of type np.array
    Return: pytorch dataloader containing the dataset of X and y that returns batches of size batch_size
    """
    cat_dict = \
    {
        'Sleep stage 1': 1, 
        'Sleep stage 2': 2, 
        'Sleep stage 3': 3, 
        'Sleep stage 4': 3,
       'Sleep stage R': 4, 
       'Sleep stage W': 0
       } if config['dataset'] == 'sleep' else \
    {
        'L_fixation': 0,
        'L_saccade': 1,
        'L_blink': 2,
        'R_fixation': 0,
        'R_saccade': 1,
        'R_blink': 2
    } 
    if config['dataset'] != 'sleep':
        func = np.vectorize(cat_dict.get)
        y = func(y)
    #y=np.vstack(y).astype(np.float32)
    y = y.astype('float32') 
    # Transform np.array to torch flaot tensor
    tensor_x = torch.as_tensor(X).float()
    tensor_y = torch.as_tensor(y).float()
    # Log the shapes
    logging.info(f"Tensor x size: {tensor_x.size()}")
    logging.info(f"Tensor y size: {tensor_y.size()}")
    # Set device
    if len(tensor_y.size()) > 2:
        tensor_y = tensor_y.squeeze()
    length, _ = tensor_y.size()
    arr = np.ones(length)
    only_fix_weight = 0.4
    blk_sac_weight = 1.0
    blk_weight = 1.0
    # print(f"only fix weight: {only_fix_weight}, blk/sacc weigth: {blk_sac_weight}")
    mask = np.apply_along_axis(lambda x: np.count_nonzero(x == 0) < 0.9 * seq_length, -1, tensor_y)
    arr[mask] = blk_sac_weight
    mask1 = np.apply_along_axis(lambda x: np.count_nonzero(x == 0) >= 0.9 * seq_length, -1, tensor_y)
    arr[mask1] = only_fix_weight
    weights = torch.DoubleTensor(arr.tolist())
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Create dataset and dataloader
    sampler = WeightedRandomSampler(weights, length)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, num_workers=4, sampler=sampler)
