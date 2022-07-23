import torch


def box_cxw_to_xlxh(x):
    cx, w = x[:, 0], x[:, 1]
    x_low = cx - 0.5 * w
    x_high = cx + 0.5 * w
    return torch.stack([x_low, x_high], dim=-1)

def box_xlxh_to_cxw(x):
    x_low, x_high = x[:, 0], x[:, 1]
    cx = (x_low + x_high)/2
    w = x_high - x_low
    return torch.stack([cx, w], dim=-1)


def time_area(boxes):
    return boxes[:, 1]-boxes[:, 0]

# modified from torchvision to also return the union
def time_iou(boxes1, boxes2):
    area1 = time_area(boxes1)
    area2 = time_area(boxes2)

    lt = torch.max(boxes1[:, None,  0], boxes2[:, 0])  # [N,M,1]
    rb = torch.min(boxes1[:, None, 1], boxes2[:, 1])  # [N,M,1]

    wh = (rb - lt).clamp(min=0)  # [N,M,1]
    inter = wh[:, :] # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_time_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, x1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 1] >= boxes1[:, 0]).all()
    assert (boxes2[:, 1] >= boxes2[:, 0]).all()

    iou, union = time_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    rb = torch.max(boxes1[:, None, 1], boxes2[:, 1])

    wh = (rb - lt).clamp(min=0)  # [N,M]
    area = wh[:, :]

    return iou - (area - union) / area