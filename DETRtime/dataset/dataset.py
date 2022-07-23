from __future__ import annotations

import logging

import torch
import torch.utils.data
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from util.box_ops import box_cxw_to_xlxh, box_xlxh_to_cxw



def unbiassed_weights(tensor_y, num_boxes=None, max_queries=10):
    """
    :param tensor_y: (samples, seq length, ) np array
    :return: weights: DoubleTensor with sampling weights
    """
    #seq_length = len(tensor_y[0])
    #logging.info("Output seq_length: {}".format(seq_length))
    num_samples = len(tensor_y)
    arr = np.ones(num_samples)

    if num_boxes is not None:
        logging.info("Setting samples > max_queries to 0")
        arr[num_boxes > max_queries] = 0
        logging.info(f'Number of elements over queries {np.sum(num_boxes > max_queries)}')
    weights = torch.DoubleTensor(arr.tolist())
    return weights

def biased_weights(tensor_y, validation=False, num_boxes=None, max_queries=10):
    """
    creates weight tensor biased towards minority classes
    :param tensor_y: (samples, seq length, ) np array
    :return: weights: DoubleTensor with sampling weights
    """
    #seq_length = len(tensor_y[0])
    num_samples = len(tensor_y)
    arr = np.ones(num_samples)

    if not validation:
        """
            follow the simple strategy of assigning higher weights to samples 
            containing a certain amount of minority samples 
        """
        f1 = lambda x: np.count_nonzero(x == 1) > 40
        #mask = np.apply_along_axis(f1, -1, tensor_y)
        mask = np.array([f1(x) for x in tensor_y])
        logging.info(f'# of time steps with saccades > 40: {np.count_nonzero(mask)}')
        arr[mask] = 2

        f2 = lambda x: np.count_nonzero(x == 2) > 20
        #mask1 = np.apply_along_axis(f2,-1, tensor_y)
        mask1 = np.array([f2(x) for x in tensor_y])
        logging.info(f'# of samples time steps > 20 blinks: {np.count_nonzero(mask1)}')
        arr[mask1] = 4

    if num_boxes is not None:
        arr[num_boxes > max_queries] = 0
        logging.info(f'# of elements with more regions than available queries {np.sum(num_boxes > max_queries)}')

    weights = torch.DoubleTensor(arr.tolist())
    return weights


def get_num_boxes(y, num_classes):
    """
    iterates over y and creates box annotations
    :param y: (seq length, ) np array
    :return: dict containing target values
    """
    i = 0
    end = len(y)
    num_boxes = 0
    class_counts = np.zeros(num_classes)
    while i < end:
        label = y[i]

        while y[i] == label and i < end - 1:  # extend event
            i += 1
        """
        for 2 classes:
        if label != 1:  # skip saccades
            num_boxes += 1
        """
        class_counts[label] += 1
        num_boxes += 1

        i += 1

    return num_boxes, class_counts


def collate_boxes(y, num_classes):
    # num_boxes = np.apply_along_axis(get_num_boxes, -1, y)
    num_boxes = []
    class_counts = np.zeros(num_classes)
    for y_i in y:
        num_boxes_i, class_counts_i = get_num_boxes(y_i, num_classes)
        class_counts += class_counts_i
        num_boxes.append(num_boxes_i)
    num_boxes = np.array(num_boxes)
    return num_boxes, class_counts


def create_annotations(y):
    """
    iterates over y and creates box annotations
    :param y: (seq length, ) np array
    :return: dict containing target values
    """
    i = 0
    end = len(y)
    bboxes = []
    labels = []
    while i < end:
        left = i
        label = y[i]

        while i < end and y[i] == label:  # extend event
            i += 1
        right = i

        # create a box of the form [x_low, x_high] in normalized form

        """
        for 2 classes: 
        if label != 1: # skip saccades
            box = torch.Tensor([left / float(end), right / float(end)])
            bboxes.append(box)
            labels.append(torch.tensor(0.) if label == 0 else torch.tensor(1))
        """
        box = torch.Tensor([left / float(end), right / float(end)])
        bboxes.append(box)
        labels.append(torch.tensor(label))  # if label == 0 else torch.tensor(1))

    if len(bboxes) > 0:
        # stack boxes in a tensor and convert boxes to [center, width]
        boxes = torch.stack(bboxes)
        boxes = box_xlxh_to_cxw(boxes)
    else:
        boxes = torch.Tensor()

    return {'boxes': boxes, 'labels': torch.as_tensor(labels)}


# def collate_fn(y):
#    labels = np.apply_along_axis(create_annotations, -1, y)
#    return labels


class TensorListDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        """
        :param X: iterable implementing __get__
        :param y: iterable implementing __get__
        """
        super(TensorListDataset, self).__init__()
        self.X = X
        self.y = y
        self.length = X.size(0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        X_idx = self.X[idx]
        annotations = create_annotations(self.y[idx])
        # boxes = annotations['boxes']
        # labels = annotations['labels']
        return X_idx.T, annotations  # {'X':X_idx, 'boxes':boxes, 'labels':labels}


def create_dataloader(data_dir, file, validation=False,
                      batch_size=32, workers=1, collate_fn=None, standard_scale=False,
                      scaler=None, max_queries=10, num_classes=3, sleep=False, apply_label_dict=False):

    """
    creates the dataloader for the DETRtime pipeline
    currently handles SleepEDF and EEG eye movement segmentation datasets
    :param collate_fn: merges a list of samples to form a batch
    :param apply_label_dict:
    :param sleep: boolean
    :param num_classes: int
    :param max_queries: int
    :param scaler: implementing sklearn.Transformer API
    :param validation: bool
    :param data_dir: str
    :param file: str
    :param batch_size: int
    :param workers: int
    :return:
    """
    logging.info("Loading data")
    data = np.load(data_dir + '/' + file)
    logging.info("Finished loading file")
    # load and process data
    logging.info("Preprocessing data")
    X = data['EEG'].astype(np.float)
    if standard_scale:
        logging.info("Rescaling data")
        if not validation and scaler is not None:
            ls, seq_length, channel = X.shape
            X = X.reshape(-1, channel)
            X = scaler.fit_transform(X)
            X = X.reshape(-1, seq_length, channel)
        elif scaler is not None:
            ls, seq_length, channel = X.shape
            X = X.reshape(-1, channel)
            X = scaler.transform(X)
            X = X.reshape(-1, seq_length, channel)

    logging.info("Annotating labels")
    if sleep:
        logging.info("Using sleep annotations")
        conv = {'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 3, 'Sleep stage 4': 3,
                'Sleep stage R': 4, 'Sleep stage W': 0}
    else:
        logging.info("Using normal annotations")
        conv = {'L_fixation': 0, 'L_saccade': 1, 'L_blink': 2, 'R_fixation': 0, 'R_saccade': 1, 'R_blink': 2}
    func = np.vectorize(conv.get, otypes=['int'])
    if apply_label_dict:
        logging.info("Applying labeling dict")
        y = func(data['labels'])
    else:
        logging.info("Not applying labeling dict")
        y = data['labels']

    # if not sleep:
    #     y = func(y)

    y = y.astype('int')
    #logging.info(f'Unique labels {np.unique(y)}')
    y = torch.as_tensor(y)
    tensor_x = torch.as_tensor(X).float()

    #Calculating number of regions within each sample
    # num_boxes, class_counts = collate_boxes(y, num_classes)  # dict collation
    # for i, c in enumerate(class_counts):
    #     logging.info(f'Class {i}: #{c} boxes')
    # # of regions preprocessed
    #logging.info("Getting num_boxes")
    #num_boxes = data['num_boxes']
    # # of boxes not used
    num_boxes = None

    dataset = TensorListDataset(tensor_x, y)

    if not validation:
        logging.info("Creating training dataloader")
        if sleep:
            weights = unbiassed_weights(y, num_boxes=num_boxes, max_queries=max_queries)
        else:
            weights = biased_weights(y, num_boxes=num_boxes, max_queries=max_queries)
        sampler = WeightedRandomSampler(weights, len(weights))
        return scaler, torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                                   num_workers=workers, collate_fn=collate_fn, drop_last=True)
    else:
        logging.info("Creating validation dataloader")
        #Generally no biased sampling of any sort in validation
        # if sleep:
        #     weights = unbiassed_weights(y, validation=True, num_boxes=num_boxes, max_queries=max_queries)
        # else:
        #     weights = biased_weights(y, validation=True, num_boxes=num_boxes, max_queries=max_queries)
        # sampler = WeightedRandomSampler(weights, len(weights))
        return scaler, torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   num_workers=workers, collate_fn=collate_fn, drop_last=True)
