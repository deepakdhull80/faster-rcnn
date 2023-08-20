import sys
sys.path.append(".")
from typing import Tuple

import torch
import torch.nn as nn
import torchvision

from model.backbone import get_backbone_f_extractor
from model.rpn import RegionProposeNetwork

class Detector(nn.Module):
    def __init__(self, image_size, backbone_name):
        super().__init__()
        self.image_size = image_size
        self.backbone, self.f_size, self.out_size = get_backbone_f_extractor(backbone_name)

        self.rpn = RegionProposeNetwork()
        
    def forward(self, images, gt_boxes=None, gt_scores=None):
        is_eval = gt_boxes is None or gt_scores is None
        
        features = self.backbone(images)
        
        ##region proposals
        self.rpn(features)