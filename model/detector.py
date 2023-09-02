import sys
sys.path.append(".")
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from model.backbone import get_backbone_f_extractor
from model.rpn import RegionProposeNetwork

class ClassificationModule(nn.Module):
    def __init__(self, out_channels, n_classes, roi_size, hidden_dim=512, p_dropout=0.3):
        super().__init__()        
        self.roi_size = roi_size
        # hidden network
        self.avg_pool = nn.AvgPool2d(self.roi_size)
        self.fc = nn.Linear(out_channels, hidden_dim)
        self.dropout = nn.Dropout(p_dropout)

        # define classification head
        self.cls_head = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, feature_map, proposals_list, gt_classes=None):
        
        if gt_classes is None:
            mode = 'eval'
        else:
            mode = 'train'
        
        # apply roi pooling on proposals followed by avg pooling
        roi_out = torchvision.ops.roi_pool(feature_map, proposals_list, self.roi_size)
        roi_out = self.avg_pool(roi_out)
        
        # flatten the output
        roi_out = roi_out.squeeze(-1).squeeze(-1)
        
        # pass the output through the hidden network
        out = self.fc(roi_out)
        out = F.relu(self.dropout(out))
        
        # get the classification scores
        cls_scores = self.cls_head(out)
        
        if mode == 'eval':
            return cls_scores
        
        # compute cross entropy loss
        cls_loss = F.cross_entropy(cls_scores, gt_classes.long())
        
        return cls_loss

class Detector(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # self.extractor, self.feature_size, out_size = get_backbone_f_extractor(config.backbone_model_name, freeze=False)
        m = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
        self.extractor = torch.nn.Sequential(*list(m.children())[:-1*config.last_n_layer_remove])
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
        self.classifier = ClassificationModule(self.feature_size, len(config.category_list.keys()) + 1, config.roi_size)
        
    def forward(self, images, gt_boxes=None, gt_class=None):
        is_eval = gt_boxes is None
        features = self.extractor(images)
        if is_eval:
            return self.rpn.predict(features)
        
        cls_loss, reg_loss, proposals, all_box_sep, gt_pos_class = self.rpn(features, gt_boxes, gt_class)
        proposals = torch.hstack((all_box_sep.view(-1, 1), proposals[0]))
        obj_loss = self.classifier(features, proposals, gt_pos_class)
        return proposals, cls_loss, reg_loss, obj_loss