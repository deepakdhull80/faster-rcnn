from model.rpn import RegionProposeNetwork
from config import Config
from model.backbone import get_backbone_f_extractor


def test_rpn():
    
    imgs, boxes = None
    extractor, out_size, feature_size = get_backbone_f_extractor("VGG")
    rpn = RegionProposeNetwork(
        Config.image_size,
        feature_size, 
        Config.anchor_scales, 
        Config.anchor_ratios,
    )
    
    feature = extractor(imgs)
    total_rpn_loss, proposals, all_box_sep = rpn(feature, boxes)
    print(total_rpn_loss)
    print(proposals.shape)
    print(proposals[:5])
    
def test_rpn_random():
    import torch
    img = torch.randn(1, 3, Config.image_size, Config.image_size)

    boxes = torch.randn(1,6,4) * 200

    extractor, out_size, feature_size = get_backbone_f_extractor("VGG")
    rpn = RegionProposeNetwork(
        Config.image_size,
        feature_size, 
        Config.anchor_scales, 
        Config.anchor_ratios,
    )

    feature = extractor(img)
    total_rpn_loss, proposals, all_box_sep = rpn(feature, boxes)
    print(total_rpn_loss)
    print(proposals.shape)
    print(all_box_sep)
    print(proposals[:5])
    
if __name__ == "__main__":
    test_rpn_random()