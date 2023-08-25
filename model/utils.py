import os
import torch
import torchvision.ops as ops

def project_bboxes(bboxes: torch.Tensor, width_scale_factor: float, height_scale_factor: float, mode='a2p'):
    assert mode in ['a2p', 'p2a']
    
    batch_size = bboxes.size(dim=0)
    proj_bboxes = bboxes.clone().reshape(batch_size, -1, 4)
    invalid_bbox_mask = (proj_bboxes == -1) # indicating padded bboxes
    
    if mode == 'a2p':
        # activation map to pixel image
        proj_bboxes[:, :, [0, 2]] *= width_scale_factor
        proj_bboxes[:, :, [1, 3]] *= height_scale_factor
    else:
        # pixel image to activation map
        proj_bboxes[:, :, [0, 2]] /= width_scale_factor
        proj_bboxes[:, :, [1, 3]] /= height_scale_factor
        
    proj_bboxes.masked_fill_(invalid_bbox_mask, -1) # fill padded bboxes back with -1
    proj_bboxes.resize_as_(bboxes)
    
    return proj_bboxes


def iou_scores(
    base_anchor, gt_boxes
    ):
    batch_size = gt_boxes.shape[0]
    flatten_base_anchor = base_anchor.view(batch_size, -1, 4)
    iou_metric = torch.zeros(batch_size, flatten_base_anchor.shape[1], gt_boxes.shape[1])
    for ib in range(batch_size):
        _iou_metric = ops.box_iou(flatten_base_anchor[ib], gt_boxes[ib])
        iou_metric[ib, :, :] = _iou_metric
    
    return iou_metric



def calc_gt_offsets(pos_anc_coords, gt_bbox_mapping):
    pos_anc_coords = ops.box_convert(pos_anc_coords, in_fmt='xyxy', out_fmt='cxcywh')
    gt_bbox_mapping = ops.box_convert(gt_bbox_mapping, in_fmt='xyxy', out_fmt='cxcywh')

    gt_cx, gt_cy, gt_w, gt_h = gt_bbox_mapping[:, 0], gt_bbox_mapping[:, 1], gt_bbox_mapping[:, 2], gt_bbox_mapping[:, 3]
    anc_cx, anc_cy, anc_w, anc_h = pos_anc_coords[:, 0], pos_anc_coords[:, 1], pos_anc_coords[:, 2], pos_anc_coords[:, 3]
    
    def fix_zero(x):
        x[x == 0] = 1
        return x
    
    gt_w = fix_zero(gt_w)
    gt_h = fix_zero(gt_h)
    
    tx_ = (gt_cx - anc_cx)/anc_w
    ty_ = (gt_cy - anc_cy)/anc_h
    # print("gt_w", gt_w.min(), gt_w.mean(), gt_w.sum())
    # print("gt_h", gt_h.min(), gt_h.mean(), gt_h.sum())
    tw_ = torch.log(gt_w / anc_w)
    th_ = torch.log(gt_h / anc_h)

    out = torch.stack([tx_, ty_, tw_, th_], dim=-1)
    out = torch.nan_to_num(out, 0)
    return out

def generate_proposals(anchors, offsets):
   
    # change format of the anchor boxes from 'xyxy' to 'cxcywh'
    anchors = ops.box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')

    # apply offsets to anchors to create proposals
    proposals_ = torch.zeros_like(anchors)
    proposals_[:,0] = anchors[:,0] + offsets[:,0]*anchors[:,2]
    proposals_[:,1] = anchors[:,1] + offsets[:,1]*anchors[:,3]
    proposals_[:,2] = anchors[:,2] * torch.exp(offsets[:,2])
    proposals_[:,3] = anchors[:,3] * torch.exp(offsets[:,3])

    # change format of proposals back from 'cxcywh' to 'xyxy'
    proposals = ops.box_convert(proposals_, in_fmt='cxcywh', out_fmt='xyxy')

    return proposals


def load_model_checkpoint(model, checkpoint_path, map_location='cpu'):
    if not os.path.exists(checkpoint_path):
        return model
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    print(model.load_state_dict(state_dict))
    return model