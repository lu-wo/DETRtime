import numpy as np
from util.box_ops import box_cxw_to_xlxh, box_xlxh_to_cxw

def generate_sequence_targets(targets, timestamps):
    '''
    In:
        targets: {
            'boxes': list of lists with [[center, width], ... ], (shape: (#boxes, 2))
            'labels': list of classes i.e. [0, 1, 1, ...] (shape: (#boxes, ))
        }
    Out:
        np.array of shape (sample_width, )
    '''

    seq = np.zeros(timestamps)

    boxes = targets['boxes']
    if len(boxes) > 0:
        boxes = box_cxw_to_xlxh(boxes)
    else:
        return seq

    for i in range(len(boxes)):
        box = boxes[i]
        label = targets['labels'][i]
        left = int(box[0] * timestamps)
        right = int(box[1] * timestamps)
        left = max(0, int(box[0] * timestamps))
        right = min(timestamps, int(box[1] * timestamps))
        event = label.cpu().detach().numpy()
        # assert right <= timestamps
        # assert left >= 0
        # assert right >= left

        seq[left:right] = np.ones(right-left) * event

    return seq

def generate_sequence_predictions(predictions, timestamps):
    '''
    In:
        predictions: {
            'pred_boxes': list of lists with [[center, width], ... ], (shape: (N, 2))
            'pred_logits': list of classes i.e. [[c_1, c_2, ...], [c_1, c_2, ...], ...] (shape: (N, #classes))
        }
    Out:
        np.array of shape (sample_width, )
    '''

    seq = np.zeros(timestamps)
    boxes = predictions['pred_boxes']
    boxes = box_cxw_to_xlxh(boxes)
    combined = [x for x in zip(boxes, predictions['pred_logits'])] # [([x,w],[c1,...,c4]), ...]
    combined.sort(key=lambda obj: max(obj[1]))

    for i in range(len(combined)):
        label = np.argmax(combined[i][1])

        if label == len(combined[i][1]) - 1:
            continue

        box = combined[i][0]
        # event = 0 if label == 0 else 2 # for 2 class
        event = label.cpu().detach().numpy()

        left = max(0, int(box[0] * timestamps))
        right = min(timestamps, int(box[1] * timestamps))
        
        assert right <= timestamps
        assert left >= 0
        assert right >= left

        seq[left:right] = np.ones(right-left) * event

    return seq
