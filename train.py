import os
import numpy as np
from glob import glob
from PIL import Image
import json

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader

from config import Config
from model.detector import Detector
from model.utils import load_model_checkpoint, project_bboxes

# Dataset
class CocoDataset(Dataset):
    def __init__(self, image_id, config):
        self.config = config
        self.image_id = image_id
        
        # transforms
        self.transforms1 = torchvision.transforms.Compose([
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float)
        ])
    
    def transform(self, image):
        image = self.transforms1(image)
        image /= 255
        return image
    
    def __getitem__(self, key):
        _id = self.image_id[key]
        image_path = os.path.join(self.config.image_dir, _id+'.jpg')
        annotation_path = os.path.join(self.config.annotation_dir, _id+'.json')
        image = Image.open(image_path)
        # get image height and width
        iwidth, iheight = image.size
        image = image.resize((self.config.image_size, self.config.image_size))
        
        box_category = json.load(open(annotation_path, 'r'))['box_category']
        def bbox_helper(l):
            box = l[0]
            x,y,w,h = box
            return [x,y,x+w,y+h]
        bbox = list(map(bbox_helper, box_category))
        bbox = torch.tensor(bbox)
        
        # project bbox to resize image scale
        bbox = project_bboxes(
            bbox.unsqueeze(0), iwidth/self.config.image_size, iheight/self.config.image_size, mode='p2a'
        ).squeeze(0)
        
        k = max(0, self.config.max_bbox - bbox.shape[0])
        if k != 0:
            invalid_pad = torch.ones((k,4)) * -1
            bbox = torch.concat([bbox, invalid_pad])
        
        image = self.transform(image)
        
        if image.shape[0] == 1:
            # grey scale
            image = torch.concat([image, image, image], axis=0)
        
        return image, bbox

    def __len__(self):
        return len(self.image_id)

# Create Training Dataset
def create_dataset(config):
    images = glob(config.image_dir+"/")
    image_id = list(map(lambda x: x.rsplit("/",1)[-1].replace(".jpg",""), filter(lambda x: x.endswith('.jpg'),images)))
    train_image_id, val_image_id = train_test_split(image_id, train_size=0.8, random_state=32, shuffle=True)
    print(train_image_id, val_image_id)
    return train_image_id, val_image_id
    
# Dataloader
def get_dl(config, train_ids, val_ids):
    trainds = CocoDataset(train_ids, config)
    valds = CocoDataset(val_ids, config)
    
    traindl = DataLoader(trainds, batch_size=config.batch_size)
    valdl = DataLoader(valds, batch_size=config.batch_size)
    return traindl, valdl

# Model
def get_model(config, device):
    
    detector = Detector(config)
    detector = load_model_checkpoint(detector, os.path.join(config.model_save_dir, f"{config.model_name}_{config.model_version}.pt"))
    return detector.to(device)

# Loss function


# Optimizer


# Training Step
def run_step(dataloader, model, optimizer, loss_fn, train=True):
    pass


def run():
    pass


if __name__ == '__main__':
    print("Starting training")
    run()