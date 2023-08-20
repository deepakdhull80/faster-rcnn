import torch
import torch.nn.functional as F

def calc_cls_loss(conf_scores_pos, conf_scores_neg):
    gt_pos_cls = torch.ones(conf_scores_pos.shape)
    gt_neg_cls = torch.zeros(conf_scores_neg.shape)
    gt = torch.concat([gt_pos_cls, gt_neg_cls])
    pred = torch.concat([conf_scores_pos, conf_scores_neg])
    return F.binary_cross_entropy(gt, pred.sigmoid())

    
def calc_bbox_reg_loss(GT_offsets, offsets_pos):
    return F.mse_loss(GT_offsets, offsets_pos)
    diff = (GT_offsets - offsets_pos) ** 2
    return diff.mean()