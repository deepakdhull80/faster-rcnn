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
        # self.extractor, self.feature_size, out_size = get_backbone_f_extractor(config.backbone_model_name, freeze=False)
        m = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
        self.extractor = torch.nn.Sequential(*list(m.children())[:-2])
        self.extractor.eval()
        self.extractor[-2].train()
        self.extractor[-1].train()
        self.feature_size = 2048
        out_size = 16
        self.rpn = RegionProposeNetwork(
            config.image_size,
            out_size,
            self.feature_size, 
            config.anchor_scales, 
            config.anchor_ratios,
        )
        
    def forward(self, images, gt_boxes=None):
        is_eval = gt_boxes is None
        features = self.extractor(images)
        if is_eval:
            return self.rpn.predict(features)
        
        cls_loss, reg_loss, proposals, all_box_sep = self.rpn(features, gt_boxes)
        
        return proposals, cls_loss, reg_loss    