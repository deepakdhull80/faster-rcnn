import sys
sys.path.append(".")

import torch
import torch.nn as nn
import torchvision

from model.utils import project_bboxes, iou_scores, calc_gt_offsets, generate_proposals
from model.loss import calc_cls_loss, calc_bbox_reg_loss

def get_anchor_base(out_size, ratios, scales):
    # TODO:Need to optimize
    anc_x = torch.arange(out_size) + 0.5
    anc_y = torch.arange(out_size) + 0.5
    
    n_anchors = len(ratios) * len(scales)
    anc_base = torch.zeros((1, out_size, out_size, n_anchors, 4))
    for i,ix in enumerate(anc_x):
        for j,iy in enumerate(anc_y):
            anchor_box = torch.zeros(
                n_anchors, 4
            )
            c = 0
            for ratio in ratios:
                for scale in scales:
                    w = scale * ratio
                    h = scale
                    xmin = ix - w / 2
                    xmax = ix + w / 2
                    ymin = iy - h / 2
                    ymax = iy + h / 2
                    box = torch.tensor([xmin, ymin, xmax, ymax])
                    anchor_box[c, :] = box
                    c += 1
            
            anc_base[:, i, j, :, :] = torchvision.ops.clip_boxes_to_image(anchor_box, (out_size, out_size))
    return anc_base


class RegionProposeNetwork(nn.Module):
    def __init__(
        self, 
        image_size: int ,
        out_size: int,
        feature_size: int, 
        anchor_scale: list, 
        anchor_ratio: list,
        pos_threshold: int = 0.7,
        neg_threshold: int = 0.3
        ):
        super().__init__()
        self.image_size = image_size
        self.feature_size = feature_size
        self.anchor_scale = anchor_scale
        self.anchor_ratio = anchor_ratio
        self.pos_threshold, self.neg_threshold = pos_threshold, neg_threshold
        
        self.conv = nn.Conv2d(feature_size, feature_size, 3, 1)
        
        n_cls = 1 * len(self.anchor_scale) * len(self.anchor_ratio)
        self.conv_cls = nn.Conv2d(feature_size, n_cls, 1, 1)
        
        n_boxes = 4 * len(self.anchor_scale) * len(self.anchor_ratio)
        self.conv_boxes = nn.Conv2d(feature_size, n_boxes, 1, 1)
        self.anchor_box = get_anchor_base(out_size, self.anchor_ratio, self.anchor_scale)
        self.w_conf, self.w_reg = 1, 5
    
    def forward(self, features, gt_boxes=None):
        batch_size = features.shape[0]
        x = self.conv(features)
        _cls = self.conv_cls(x)
        _boxes = self.conv_boxes(x)
        self.scale_factor = self.image_size // x.shape[-1]
        # base_anchor -> (1, out_size, out_size, n_anchor, 4)
        anchor_box = self.anchor_box.clone()
        base_anchor = anchor_box.to(features.device)
        base_anchor = base_anchor.repeat(batch_size, 1, 1, 1, 1)
        # gt_boxes -> (batch_size, max_boxes, 4)
        gt_boxes = project_bboxes(gt_boxes, self.scale_factor, self.scale_factor, mode="p2a")
        
        pos_anchor_idx, pos_anchor_box, pos_gt_box, \
            all_box_sep, gt_offset, neg_anchor_idx = self.get_req_anchor(
            base_anchor, gt_boxes, self.pos_threshold, self.neg_threshold
        )
        
        pred_pos_cls = _cls.contiguous().view(-1)[pos_anchor_idx]
        pred_offset = _boxes.contiguous().view(-1, 4)[pos_anchor_idx]
        pred_neg_cls = _cls.view(-1)[neg_anchor_idx]
        proposals = generate_proposals(pos_anchor_box, pred_offset)
        ''' for loss function 
        
            we need positive gt_offset_boxes, pred_offset_boxes, gt_score, pred_score
            negative gt_score, pred_score
        '''
        cls_loss = calc_cls_loss(pred_pos_cls, pred_neg_cls)
        reg_loss = calc_bbox_reg_loss(gt_offset, pred_offset)
        total_rpn_loss = self.w_conf * cls_loss + self.w_reg * reg_loss
        return total_rpn_loss, proposals, all_box_sep
    
    def get_req_anchor(self, base_anchor, gt_boxes, pos_threshold, neg_threshold):
        
        iou_metric = iou_scores(
            base_anchor, gt_boxes
        )
        
        # positive anchor index and it's iou score 
        max_iou_per_gt_box, _ = iou_metric.max(dim=1, keepdim=True)
        mask = torch.logical_and(iou_metric == max_iou_per_gt_box, max_iou_per_gt_box > 0)
        mask = torch.logical_or(iou_metric > pos_threshold, mask)
        pos_anchor_idx, pos_box_idx = torch.where(mask.flatten(0,1))
        pos_anchor_box = base_anchor.view(-1, 4)[pos_anchor_idx]
        pos_gt_box = gt_boxes.view(-1, 4)[pos_box_idx]
        all_box_sep = torch.where(mask)[0]
        gt_offset = calc_gt_offsets(pos_anchor_box, pos_gt_box)
        
        # negative index and score
        neg_mask = iou_metric < neg_threshold
        neg_anchor_idx = torch.where(neg_mask.flatten(0, 1))[1]
        
        neg_anchor_idx = neg_anchor_idx[torch.randint(low=0, high=neg_anchor_idx.shape[0], size=(pos_anchor_idx.shape[0],))]
        return pos_anchor_idx, pos_anchor_box, pos_gt_box, all_box_sep, gt_offset, neg_anchor_idx
        
if __name__ == "__main__":
    import time
    from config import Config
    from model.backbone import get_backbone_f_extractor
    image_size = 768
    img = torch.randn(2, 3, image_size, image_size)
    
    boxes = torch.randn(2,6,4) * 200
    
    extractor,feature_size, out_size = get_backbone_f_extractor("VGG")
    
    rpn = RegionProposeNetwork(
        Config.image_size,
        out_size,
        feature_size, 
        Config.anchor_scales, 
        Config.anchor_ratios,
    )
    
    feature = extractor(img)
    st = time.time()
    
    total_rpn_loss, proposals, all_box_sep = rpn(feature, boxes)
    print(total_rpn_loss)
    print(proposals.shape)
    print(all_box_sep)
    print(proposals[:5])
    
    print(f"Time Taken: {time.time() - st:.2f} sec")