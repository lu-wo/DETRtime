"""
copied from DeTR implementation and rewritten for time series segmentation
"""
import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from util.box_ops import box_cxw_to_xlxh, generalized_time_iou

from util.misc import accuracy
import logging
from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer


class DETRtime(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries):#, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        #final predictions
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

        #outputs two values for bounding box, 3 layers
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        #CNN --> Transformer projector
        self.input_proj = nn.Conv1d(backbone.num_channels, hidden_dim, 1)
        self.backbone = backbone
        #self.aux_loss = aux_loss

    def forward(self, samples: torch.Tensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x C x S]
                     It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries (how many events are detected)
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                                dictionaries containing the two above keys for each decoder layer.
        """
        #changed to only account for tensors
        # if isinstance(samples, (list, torch.Tensor)):
        #     samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        #src, mask = features[-1].decompose()
        src = features
        #no attention mask needed
        #assert mask is not None
        mask = None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)[0]

        outputs_class = self.class_embed(hs)
        #for no reasons currently
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        # #not needed for our purposes
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    #@torch.jit.unused
    #def _set_aux_loss(self, outputs_class, outputs_coord):
    #    # this is a workaround to make torchscript happy, as torchscript
    #    # doesn't support dictionary with non-homogeneous values, such
    #    # as a dict having both a Tensor and a list.
    #    return [{'pred_logits': a, 'pred_boxes': b}
    #            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        if self.num_classes == 3: #hardcoded class weights
            #logging.info("Using adjusted class_weights for movie")
            #empty_weight = torch.tensor([0.61, 0.83, 6.59, self.eos_coef]) #calculated, but found to be worse
            #empty_weight = torch.tensor([0.61, 1.4, 6.59, self.eos_coef]) #adjusted saccades higher movie
            logging.info("Using adjusted class weights for Zuco")
            empty_weight = torch.tensor([0.64, 1.7, 9.4, self.eos_coef])#zuco
            logging.info(f'Weights: {empty_weight}')
        elif self.num_classes == 5: #hardcoded class weights for the sleep data set
            logging.info("Using adjusted class weights (events) for Sleep")
            #logging.info("No adjusted class weights for Sleep")
            logging.info("Weighted by labels")
            #empty_weight = torch.tensor([0.44, 1.25, 0.72, 2.08, 2.38, self.eos_coef])
            #empty_weight = torch.tensor([0.5, 3.6, 1.7, 6.31, 4.0, self.eos_coef])#sleep,labels, best so far
            empty_weight = torch.tensor([0.5, 4.0, 1.7, 6.31, 4.0, self.eos_coef])
            #logging.info("Weighted by events")
            #empty_weight = torch.tensor([0.94, 0.55, 0.53, 1.29, 1.95, self.eos_coef])#adjusted to events
            logging.info(f'Weights: {empty_weight}')
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o.long()

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 2]
           The target boxes are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        #################     print bbox      ######################
        #print(f'Target boxs: {target_boxes}\n Predicted boxes: {src_boxes}')
        ############################################################

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_time_iou(
            box_cxw_to_xlxh(src_boxes),
            box_cxw_to_xlxh(target_boxes)
        ))

        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it more readily corresponds to the the total number of classes + 1,
    # as an additional "dummy" class is used

    num_classes = args.num_classes
    
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETRtime(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        #aux_loss=args.aux_loss,
    )
    # if args.masks:
    #     model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    #if args.masks:
    #    weight_dict["loss_mask"] = args.mask_loss_coef
    #    weight_dict["loss_dice"] = args.dice_loss_coef

    # TODO this is a hack
    # if args.aux_loss:
    #     aux_weight_dict = {}
    #     for i in range(args.dec_layers - 1):
    #         aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    #     weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    #if args.masks:
    #    losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    #postprocessors = {'bbox': PostProcess()}
    # if args.masks:
    #     postprocessors['segm'] = PostProcessSegm()
    return model, criterion #, postprocessors