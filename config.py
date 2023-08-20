
class Config:
    image_size: int = 256
    anchor_ratios: list = [0.5, 1, 1.5] # width
    anchor_scales: list = [2, 4, 6]
    anchor_pos_threshold: int = 0.7
    anchor_neg_threshold: int = 0.3
    max_bbox: int = 32