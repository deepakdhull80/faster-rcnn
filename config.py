
class Config:
    image_size: int = 256
    anchor_ratios: list = [0.5, 1, 1.5] # width
    anchor_scales: list = [2, 4, 6]
    anchor_pos_threshold: int = 0.7
    anchor_neg_threshold: int = 0.3
    max_bbox: int = 20
    backbone_model_name: str = "VGG"
    image_dir: str = "/home/ubuntu/user-files/deepak.dhull/workspace/practice/faster-rcnn/data/images/train2017"
    annotation_dir: str = "/home/ubuntu/user-files/deepak.dhull/workspace/practice/faster-rcnn/data/annotations"
    batch_size: int = 8*5
    model_save_dir: str = "/home/ubuntu/user-files/deepak.dhull/workspace/practice/faster-rcnn/checkpoint"
    model_version: str = 'v2'
    model_name: str = "detector"
    device: str = "cuda:2"
    lr: float = 1e-2
    weight_decay: float = 0
    epoch: int = 30
    init_worker: int = 20