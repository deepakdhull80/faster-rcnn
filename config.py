
class Config:
    image_size: int = 256
    anchor_ratios: list = [0.5, 1, 1.5] # width
    anchor_scales: list = [2, 4, 6]
    anchor_pos_threshold: int = 0.7
    anchor_neg_threshold: int = 0.3
    max_bbox: int = 32
    backbone_model_name: str = "VGG"
    image_dir: str = "/home/ubuntu/user-files/deepak.dhull/workspace/practice/faster-rcnn/data/images/train2017"
    annotation_dir: str = "/home/ubuntu/user-files/deepak.dhull/workspace/practice/faster-rcnn/data/annotations"
    batch_size: int = 1
    model_save_dir: str = "/home/ubuntu/user-files/deepak.dhull/workspace/practice/faster-rcnn/checkpoint"
    model_version: str = 'v1'
    model_name: str = "detector"
    device: str = "cuda:3"
    lr: float = 1e-3
    weight_decay: float = 0
    epoch: int = 30