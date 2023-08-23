
class Config:
    image_size: int = 256
    anchor_ratios: list = [0.5, 1, 1.5] # width
    anchor_scales: list = [2, 4, 6]
    anchor_pos_threshold: int = 0.7
    anchor_neg_threshold: int = 0.3
    max_bbox: int = 32
    model_name: str = "VGG"
    image_dir: str = None
    annotation_dir: str = None
    batch_size: int = 32
    model_save_dir: str = None
    model_version: str = 'v1'
    model_name: str = "detector"
    device: str = "cpu"