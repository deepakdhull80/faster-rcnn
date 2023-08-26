import os
import numpy as np
from glob import glob
from PIL import Image
import json

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import Config
from model.detector import Detector
from model.utils import load_model_checkpoint, project_bboxes

# torch.autograd.set_detect_anomaly(True)
 
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
        # print(_id, bbox)
        # project bbox to resize image scale
        bbox = project_bboxes(
            bbox.unsqueeze(0), iwidth/self.config.image_size, iheight/self.config.image_size, mode='p2a'
        ).squeeze(0)
        
        k = max(0, self.config.max_bbox - bbox.shape[0])
        if k != 0:
            invalid_pad = torch.ones((k,4)) * -1
            bbox = torch.concat([bbox, invalid_pad])
        
        bbox = bbox[:self.config.max_bbox,:]
        
        image = self.transform(image)
        
        if image.shape[0] == 1:
            # grey scale
            image = torch.concat([image, image, image], axis=0)
        
        return image, bbox

    def __len__(self):
        return len(self.image_id)

# Create Training Dataset
def create_dataset(config):
    images = glob(config.image_dir+"/*jpg")
    image_id = list(map(lambda x: x.rsplit("/",1)[-1].replace(".jpg",""), filter(lambda x: x.endswith('.jpg'),images)))
    
    # annotation avalable
    anno = glob(config.annotation_dir+"/*.json")
    anno_id = list(map(lambda x: x.rsplit("/",1)[-1].replace(".json",""), filter(lambda x: x.endswith('.json'),anno)))
    
    image_id = list(set(image_id).intersection(anno_id))
    
    train_image_id, val_image_id = train_test_split(image_id, train_size=0.8, random_state=32, shuffle=True)
    print(len(train_image_id), len(val_image_id))
    return train_image_id, val_image_id
    
# Dataloader
def get_dl(config, train_ids, val_ids):
    trainds = CocoDataset(train_ids, config)
    valds = CocoDataset(val_ids, config)
    
    traindl = DataLoader(trainds, batch_size=config.batch_size, num_workers=config.init_worker)
    valdl = DataLoader(valds, batch_size=config.batch_size, num_workers=config.init_worker)
    return traindl, valdl

# Model
def get_model(config, device):
    
    detector = Detector(config)
    try:
        detector = load_model_checkpoint(detector, os.path.join(config.model_save_dir, f"{config.model_name}_{config.model_version}.pt"))
    except Exception as e:
        print(f"[Failed] model checkpoint not able to load. {e}")
    return detector.to(device)

# Loss function
# loss is mse + bce computed in model.rpn itself

# Optimizer
def get_optim(model, lr=1e-3, weight_decay=1e-6):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Training Step
def run_step(epoch, dataloader, model, optimizer, train=True, device=torch.device("cpu")):
    itr = tqdm(dataloader, total=len(dataloader))
    model = model.train(train)
    total_loss = 0
    mode = "train" if train else "eval"
    for i, batch in enumerate(itr):
        optimizer.zero_grad()
        images, bbox = batch[0].to(device), batch[1].to(device)
        proposal, loss = model(images, bbox)
        if train:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            total_loss += loss.item()
        itr.set_description(f"Epoch({mode}): {epoch}, Loss:{loss.item():.5f}, Avg Loss:{total_loss/(1+i) :.5f}")
    return total_loss / len(dataloader)
    
    
def run(writer=None):
    train, val = create_dataset(Config)
    train_dl, val_dl = get_dl(Config, train, val)
    device = torch.device(Config.device)
    model = get_model(Config, device)
    optim = get_optim(model,Config.lr, Config.weight_decay)
    glob_loss = 1e3
    for epoch in range(Config.epoch):
        train_loss = run_step(epoch, train_dl, model, optim, device=device)
        val_loss = run_step(epoch, val_dl, model, optim, train=False, device=device)
        
        if writer:
            writer.add_scalar('training loss',
                                train_loss,
                                epoch+1)
            
            writer.add_scalar('Validation loss',
                                val_loss,
                                epoch+1)
        if val_loss < glob_loss:
            glob_loss = val_loss
            # save checkpoint
            torch.save(model.state_dict(), os.path.join(Config.model_save_dir, f"{Config.model_name}_{Config.model_version}.pt"))
            print("Checkpoint saved")
if __name__ == '__main__':
    print("Starting training")
    writer = SummaryWriter('logs/faster-rcnn')
    run(writer)