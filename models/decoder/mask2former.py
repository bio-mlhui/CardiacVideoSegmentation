# multi-scale features, b c h w -> module -> obj queries, predictions, b nq c
import torch.nn as nn
from models.layers.decoder_layers import CrossAttentionLayer, SelfAttentionLayer, FFNLayer
from models.layers.anyc_trans import MLP
import torch.nn.functional as F
import torch
import copy
from models.layers.utils import zero_module, _get_clones
from models.layers.position_encoding import build_position_encoding
from einops import rearrange, reduce, repeat
from scipy.optimize import linear_sum_assignment
from models.layers.matching import batch_dice_loss, batch_sigmoid_ce_loss, batch_sigmoid_focal_loss, dice_loss, ce_mask_loss
from detectron2.modeling import META_ARCH_REGISTRY
import detectron2.utils.comm as comm
import data_schedule.utils.box_ops as box_ops
from data_schedule.utils.segmentation import small_object_weighting
from models.layers.utils import zero_module
from utils.misc import is_dist_avail_and_initialized
from collections import defaultdict
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from torch.cuda.amp import autocast

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))

class Image_SetMatchingLoss(nn.Module):
    def __init__(self, 
                 loss_config,
                 num_classes,) -> None:
        super().__init__()
        self.num_classes = num_classes # n=1 / n=0 / n>1
        self.matching_metrics = loss_config['matching_metrics'] # mask: mask/dice; point_sample_mask: ..
        self.losses = loss_config['losses'] 

        self.aux_layer_weights = loss_config['aux_layer_weights']  # int/list
        if 'class_ce' in self.losses:
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = loss_config['background_cls_eos']
            self.register_buffer('empty_weight', empty_weight)
        # self.register_buffer('small_obj_weight', torch.tensor(loss_config['small_obj_weight']).float())

    @property
    def device(self,):
        return self.empty_weight.device
    
    def compute_loss(self, model_outs, targets):
        if 'masks' in targets:
            num_objs = torch.stack(targets['masks'], dim=0).flatten(1).any(-1).int().sum() 
        elif 'boxes' in targets:
            gt_boxes = torch.stack(targets['boxes'], dim=0) # bn 4
            num_objs = (gt_boxes[:, 2:] > 0).all(-1).int().sum()
        else:
            raise ValueError('targets里没有boxes/masks, 需要确定数量')
        
        num_objs = torch.as_tensor([num_objs], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objs)
        num_objs = torch.clamp(num_objs / comm.get_world_size(), min=1).item()
        if isinstance(self.aux_layer_weights, list):
            assert len(self.aux_layer_weights) == (len(model_outs) - 1)
        else:
            self.aux_layer_weights = [self.aux_layer_weights] * (len(model_outs) - 1)

        layer_weights = self.aux_layer_weights + [1.]

        loss_values = {
            'mask_dice':0., 'mask_ce':0.,
            'box_l1': 0., 'box_giou': 0.,
            'class_ce':0.,

            'mask_dice_smobj':0., 'mask_ce_smobj':0.,
            'boxMask_dice':0., 'boxMask_ce':0.,
        }
        
        if ('mask_ce_dice' in self.matching_metrics) or ('mask_ce_dice' in self.losses):
            # mask interpolate
            tgt_mask_shape = targets['masks'][0].shape[-2:] # list[n H W], b
            for layer_idx in range(len(model_outs)):
                # b nq h w
                model_outs[layer_idx]['pred_masks'] = F.interpolate(model_outs[layer_idx]['pred_masks'],
                                                                    size=tgt_mask_shape, mode='bilinear', align_corners=False)
        for taylor, layer_out in zip(layer_weights, model_outs):
            if taylor != 0:
                matching_indices = self.matching(layer_out, targets)
                for loss in self.losses:
                    loss_extra_param = self.losses[loss]
                    if loss == 'mask_dice_ce' :
                        loss_dict = self.loss_mask_dice_ce(layer_out, targets, matching_indices, num_objs,
                                                           loss_extra_param=loss_extra_param)
                    elif loss == 'boxMask_dice_ce':
                        pass
                    elif loss == 'box_l1_giou':
                        loss_dict = self.loss_box_l1_giou(layer_out, targets, matching_indices, num_objs,
                                                          loss_extra_param=loss_extra_param)
                    elif loss == 'class_ce':
                        loss_dict = self.loss_class_ce(layer_out, targets, matching_indices, num_objs,
                                                       loss_extra_param=loss_extra_param)
                    elif loss == 'point_mask_dice_ce':
                        loss_dict = self.loss_point_mask_dice_ce(layer_out, targets, matching_indices, num_objs,
                                                                 loss_extra_param=loss_extra_param)
                    else:
                        raise ValueError()
                    
                    for key, value in loss_dict.items():
                        loss_values[key] = loss_values[key] + value
    
        return loss_values      

    @torch.no_grad()
    def matching(self, layer_out, targets):
        batch_size = len(targets['masks']) if 'masks' in targets else len(targets['boxes'])
        indices = [] 
        for i in range(batch_size):
            C = 0.

            if 'class_prob' in self.matching_metrics:
                out_cls = layer_out['pred_class'][i].softmax(-1) # nq c
                tgt_cls = targets['classes'][i] # n
                cost_class = - out_cls[:, tgt_cls] # nq n
                C += self.matching_metrics['class_prob']['prob'] * cost_class

            if 'mask_dice_ce' in self.matching_metrics:
                out_mask = layer_out['pred_masks'][i]  # nq h w
                tgt_mask = targets['masks'][i].to(out_mask) # ni H W
                cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) 
                cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1))
                C += self.matching_metrics['mask_dice_ce']['ce'] * cost_mask + \
                     self.matching_metrics['mask_dice_ce']['dice'] * cost_dice

            if 'box_l1_giou' in self.matching_metrics:
                out_box = layer_out['pred_boxes'][i].sigmoid() # nq 4
                tgt_bbox = targets['boxes'][i] # ni 4 
                cost_l1 = torch.cdist(out_box, tgt_bbox, p=1) 
                cost_giou = 1 - box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(out_box),
                                                        box_ops.box_cxcywh_to_xyxy(tgt_bbox))
                C += self.matching_metrics['box_l1_giou']['l1'] * cost_l1 + \
                      self.matching_metrics['box_l1_giou']['giou'] + cost_giou
 
            if 'point_mask_dice_ce' in self.matching_metrics:
                out_mask = layer_out['pred_masks'][i]  # nq h w
                tgt_mask = targets['masks'][i].to(out_mask) # ni H W

                out_mask = out_mask[:, None]
                tgt_mask = tgt_mask[:, None]
                # all masks share the same set of points for efficient matching!
                point_coords = torch.rand(1, self.matching_metrics['point_mask_dice_ce']['num_points'],
                                           2, device=self.device)
                # get gt labels
                tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                with autocast(enabled=False):
                    out_mask = out_mask.float() # nq num_points
                    tgt_mask = tgt_mask.float()
                    cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                    cost_dice = batch_dice_loss(out_mask, tgt_mask)
                C += self.matching_metrics['point_mask_dice_ce']['ce'] * cost_mask + \
                     self.matching_metrics['point_mask_dice_ce']['dice'] * cost_dice
            
            C = C.cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def loss_mask_dice_ce(self, outputs, targets, indices, num_objs, loss_extra_param):
        src_masks = outputs['pred_masks'] # b nq h w
        tgt_masks = targets['masks']
        src_masks = torch.cat([t[J] for t, (J, _) in zip(src_masks, indices)], dim=0)
        tgt_masks = torch.cat([t[J] for t, (_, J) in zip(tgt_masks, indices)], dim=0)
        tgt_masks = tgt_masks.to(src_masks)
        losses = {
            "mask_ce": ce_mask_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_objs),
            "mask_dice": dice_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_objs),
        }
        return losses   

    def loss_point_mask_dice_ce(self, outputs, targets, indices, num_objs, loss_extra_param):
        src_masks = outputs['pred_masks'] # b nq h w
        tgt_masks = targets['masks']
        src_masks = torch.cat([t[J] for t, (J, _) in zip(src_masks, indices)], dim=0) # n_sigma h w
        tgt_masks = torch.cat([t[J] for t, (_, J) in zip(tgt_masks, indices)], dim=0) # n_sigma h w
        tgt_masks = tgt_masks.to(src_masks)

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = tgt_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                loss_extra_param['num_points'],
                loss_extra_param['oversample_ratio'],
                loss_extra_param['importance_sample_ratio'],
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "mask_dice": ce_mask_loss(point_logits, point_labels, num_objs),
            "mask_ce": dice_loss(point_logits, point_labels, num_objs),
        }

        del src_masks
        del target_masks
        return losses        

    def loss_class_ce(self, outputs, targets, indices, num_objs, loss_extra_param):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs["pred_class"].float() # b nq c

        idx = self._get_src_permutation_idx(indices)
        # list[n], b -> bn
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets['classes'], indices)]) 
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=self.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"class_ce": loss_ce}
        return losses

    def loss_box_l1_giou(self, outputs, targets, indices, num_objs, loss_extra_param): 
        tgt_boxes = targets['boxes'] # list[n 4], b

        src_boxes = outputs['pred_boxes'].sigmoid() # b nq 4

        src_boxes = torch.cat([t[J] for t, (J, _) in zip(src_boxes, indices)], dim=0) # n_sum 4
        tgt_boxes = torch.cat([t[J] for t, (_, J) in zip(tgt_boxes, indices)], dim=0) # n_sum 4
        
        loss_l1 = F.l1_loss(src_boxes, tgt_boxes, reduction='none') # n_sum 4

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                                    box_ops.box_cxcywh_to_xyxy(tgt_boxes)))
        return {
            'box_l1': loss_l1.sum(-1).sum() / num_objs,
            'box_giou': loss_giou.sum() / num_objs
        }

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


@META_ARCH_REGISTRY.register()
class Image_MaskedAttn_MultiscaleMaskDecoder(nn.Module):
    def __init__(self,
                 configs,
                 multiscale_shapes):
        super().__init__()
        d_model = configs['d_model']
        attn_configs = configs['attn']
        self.nqueries = configs['nqueries']
        self.nlayers = configs['nlayers']
        self.memory_scales = configs['memory_scales']
        self.mask_scale = configs['mask_scale']
        self.mask_spatial_stride = multiscale_shapes[self.mask_scale].spatial_stride
        num_classes = configs['num_classes']

        inputs_projs = configs['inputs_projs']
        self.inputs_projs = nn.Sequential()
        if inputs_projs is not None:
            self.inputs_projs = META_ARCH_REGISTRY.get(inputs_projs['name'])(inputs_projs, 
                                                                             multiscale_shapes=multiscale_shapes,
                                                                             out_dim=d_model)
        self.query_poses =  nn.Embedding(self.nqueries, d_model)
        self.query_feats = nn.Embedding(self.nqueries, d_model)
        self.level_embeds = nn.Embedding(len(self.memory_scales), d_model)
        assert self.nlayers % len(self.memory_scales) == 0
        self.cross_layers = _get_clones(CrossAttentionLayer(d_model=d_model,
                                                            nhead=attn_configs['nheads'],
                                                            dropout=0.0,
                                                            normalize_before=attn_configs['normalize_before']),
                                        self.nlayers)
        self.self_layers = _get_clones(SelfAttentionLayer(d_model=d_model,
                                                          nhead=attn_configs['nheads'],
                                                          dropout=0.0,
                                                          normalize_before=attn_configs['normalize_before']),
                                       self.nlayers)  
        self.ffn_layers = _get_clones(FFNLayer(d_model=d_model,
                                               dim_feedforward=attn_configs['dim_feedforward'],
                                               dropout=0.0,
                                               normalize_before=attn_configs['normalize_before']),
                                      self.nlayers) 
                  
        self.nheads = attn_configs['nheads']
        self.query_norm = nn.LayerNorm(d_model)
        self.pos_2d = build_position_encoding(position_embedding_name='2d')
        if 'loss' in configs:
            self.loss_module = Image_SetMatchingLoss(loss_config=configs['loss'], num_classes=num_classes)
        
        self.head_outputs = configs['head_outputs']
        
        assert 'mask' in self.head_outputs
        self.query_mask = MLP(d_model, d_model, d_model, 3)
        if 'class' in self.head_outputs:
            self.query_class = nn.Linear(d_model, num_classes + 1)
        if 'box' in self.head_outputs:
            self.query_box = MLP(d_model, d_model, 4, 3)
        # box_mask?
        # polygn
        
    @property
    def device(self,):
        return self.query_feats.weight.device

    def get_memories_and_mask_features(self, multiscales):
        # b c h w -> hw b c
        memories = [multiscales[scale] for scale in self.memory_scales]
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories_poses = [self.pos_2d(torch.zeros_like(mem)[:, 0, :, :].bool(), hidden_dim=mem.shape[1]) for mem in memories]
        memories = [rearrange(mem, 'b c h w -> (h w) b c') for mem in memories]
        memories_poses = [rearrange(mem_pos, 'b c h w -> (h w) b c') for mem_pos in memories_poses]
        mask_features = multiscales[self.mask_scale] # b c h w
        return memories, memories_poses, mask_features, size_list

    def forward(self, 
                multiscales, # b c h w
                ):
        multiscales = self.inputs_projs(multiscales)
        # hw b c; b c h w
        memories, memories_poses, mask_features, size_list = self.get_memories_and_mask_features(multiscales)
        memories = [mem_feat + self.level_embeds.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]
        batch_size = mask_features.shape[0]
        query_poses = self.query_poses.weight.unsqueeze(1).repeat(1, batch_size, 1)
        output = self.query_feats.weight.unsqueeze(1).repeat(1, batch_size, 1)

        ret = []
        outputs_class, outputs_box, outputs_mask, attn_mask, queries = \
            self.forward_heads(output, mask_features, attn_mask_target_size=size_list[0])
        ret.append({'pred_class':outputs_class, 'pred_masks': outputs_mask, 'pred_boxes': outputs_box, 'queries': queries})

        for i in range(self.nlayers):
            level_index = i % len(self.memory_scales)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False  # 全masked掉的 全注意, 比如有padding
            output = self.cross_layers[i](
                tgt=output,
                memory=memories[level_index], 
                memory_mask=attn_mask, 
                memory_key_padding_mask=None,
                pos=memories_poses[level_index], 
                query_pos=query_poses,
            )
            output = self.self_layers[i](
                output,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_poses,
            )
            output = self.ffn_layers[i](
                output 
            )
            outputs_class, outputs_box, outputs_mask, attn_mask, queries = \
                self.forward_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % len(self.memory_scales)])
            ret.append({'pred_class':outputs_class, 'pred_masks': outputs_mask, 'pred_boxes': outputs_box, 'queries': queries})

        assert len(ret) == self.nlayers + 1        
        return ret

    def forward_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.query_norm(output)
        decoder_output = decoder_output.transpose(0, 1).contiguous()

        box_logits = self.query_box(decoder_output) if 'box' in self.head_outputs else None # b n 4
        class_logits = self.query_class(decoder_output) if 'class' in self.head_outputs else None # b n class+1
        mask_embeds = self.query_mask(decoder_output)  # b n c
        mask_logits = torch.einsum("bqc,bchw->bqhw", mask_embeds, mask_features)  

        # b nq h w
        attn_mask = mask_logits.detach().clone() 
        attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # b*head nq hw
        attn_mask = (attn_mask.flatten(2).unsqueeze(1).repeat(1, self.nheads, 1, 1).flatten(0, 1).sigmoid() < 0.5).bool()
        
        mask_logits = F.interpolate(mask_logits, scale_factor=self.mask_spatial_stride, mode='bilinear', align_corners=False)

        if self.training:
            return class_logits, box_logits, mask_logits, attn_mask, decoder_output # b nq h w
        else:
            return class_logits.softmax(-1) if class_logits is not None else None,\
                  box_logits.sigmoid() if box_logits is not None else None, mask_logits, attn_mask, decoder_output # b nq h w
    

    def compute_loss(self, outputs, targets):
        assert self.training
        return self.loss_module.compute_loss(outputs, targets)

@META_ARCH_REGISTRY.register()
class Video2D_Image_MaskedAttn_MultiscaleMaskDecoder(nn.Module):
    def __init__(self,
                 configs,
                 multiscale_shapes,
                 ):
        super().__init__()
        self.image_homo = Image_MaskedAttn_MultiscaleMaskDecoder(configs=configs,
                                                                 multiscale_shapes=multiscale_shapes,)
    
    def forward(self, multiscales): # b c t h w
        batch_size, _, nf = multiscales[list(multiscales.keys())[0]].shape[:3]
        # b c t h w -> bt c h w
        multiscales = {key: value.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous() for key,value in multiscales.items()}

        predictions = self.image_homo(multiscales) # bt nq class; bt nq h w; bt nq 
        
        for layer_idx in range(len(predictions)):
            if 'box' in self.image_homo.head_outputs:
                predictions[layer_idx]['pred_boxes'] = rearrange(predictions[layer_idx]['pred_boxes'], '(b t) nq c -> b t nq c',b=batch_size, t=nf)
            if 'class' in self.image_homo.head_outputs:
                predictions[layer_idx]['pred_class'] = rearrange(predictions[layer_idx]['pred_class'],'(b t) nq c -> b t nq c',b=batch_size, t=nf)
            predictions[layer_idx]['pred_masks'] = rearrange(predictions[layer_idx]['pred_masks'], '(b t) nq h w -> b t nq h w',b=batch_size, t=nf)
            predictions[layer_idx]['queries'] = rearrange(predictions[layer_idx]['queries'], '(b t) nq c -> b t nq c',b=batch_size, t=nf)
        return predictions      
    
    def compute_loss(self, predictions, targets, frame_targets):
        has_ann = targets['has_ann'].flatten() # bT
        for layer_idx in range(len(predictions)):
            predictions[layer_idx]['pred_boxes'] = predictions[layer_idx]['pred_boxes'].flatten(0,1)[has_ann].contiguous() if 'box' in self.image_homo.head_outputs else None
            predictions[layer_idx]['pred_class'] = predictions[layer_idx]['pred_class'].flatten(0,1)[has_ann].contiguous() if 'class' in self.image_homo.head_outputs else None
            # bt nq h w
            predictions[layer_idx]['pred_masks'] = predictions[layer_idx]['pred_masks'].flatten(0,1)[has_ann].contiguous()
        return self.image_homo.compute_loss(predictions, frame_targets)
    