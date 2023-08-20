import sys
sys.path.append(".")
from typing import Tuple

import torch
import torch.nn as nn
import torchvision

from config import Config

class BackboneName:
    VGG: str = "vgg16"
    RESNET: str = ""

feature_extractor_mapper = dict(
    VGG={"name":'features', "n_layers": 8},
    RESNET={"name":'features', "n_layers": 8}
)

def get_backbone_f_extractor(backbone_name: str, pretrained=True, freeze=False) -> Tuple[nn.Module, int, int]:
    name = BackboneName.__dict__[backbone_name]
    model = getattr(torchvision.models, name)(pretrained=pretrained)
    
    x = torch.randn(1,3,Config.image_size, Config.image_size)
    extractor = getattr(model, feature_extractor_mapper[backbone_name]['name'])
    
    # get require layers
    req_layers = list(extractor.children())[:feature_extractor_mapper[backbone_name]['n_layers']]
    extractor = nn.Sequential(*req_layers)
    if not freeze:
        for param in extractor.named_parameters():
            param[1].requires_grad = True
    
    x = extractor(x)
    return extractor, x.shape[1], x.shape[-1]

if __name__ == "__main__":
    extractor, out_size, feature_size = get_backbone_f_extractor("VGG")
    print(out_size)
    print(extractor)