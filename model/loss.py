import torch
import torch.nn.functional as F

bceloss = torch.nn.BCEWithLogitsLoss()


def calc_cls_loss(conf_scores_pos, conf_scores_neg):
    gt_pos_cls = torch.ones(conf_scores_pos.shape).to(conf_scores_pos.device)
    gt_neg_cls = torch.zeros(conf_scores_neg.shape).to(conf_scores_pos.device)
    gt = torch.concat([gt_pos_cls, gt_neg_cls])
    pred = torch.concat([conf_scores_pos, conf_scores_neg])
    return bceloss(pred, gt)

    
def calc_bbox_reg_loss(GT_offsets, offsets_pos):
    # print("GT_offsets, offsets_pos",GT_offsets, offsets_pos)
    return F.mse_loss(GT_offsets, offsets_pos)