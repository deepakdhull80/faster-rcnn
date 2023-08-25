import sys
sys.path.append(".")
from typing import Tuple

import torch
import torch.nn as nn
import torchvision

from model.backbone import get_backbone_f_extractor
from model.rpn import RegionProposeNetwork

class Detector(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.extractor, out_size, self.feature_size = get_backbone_f_extractor(config.backbone_model_name)
        self.rpn = RegionProposeNetwork(
            config.image_size,
            self.feature_size, 
            config.anchor_scales, 
            config.anchor_ratios,
        )
        
    def forward(self, images, gt_boxes=None):
        is_eval = gt_boxes is None
        features = self.extractor(images)
        if is_eval:
            raise NotImplementedError("Prediction is not implemented")
        
        total_rpn_loss, proposals, all_box_sep = self.rpn(features, gt_boxes)
        
        return proposals, total_rpn_loss