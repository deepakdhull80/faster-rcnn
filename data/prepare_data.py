'''
Prepare data as require for training
    - data/images -> images with unique file names
    - data/annotations -> annotation file for an image in specific structure
        - annotation file structure
'''
import json
import zipfile
import os
import requests
import sys
import json
import pandas as pd
from clint.textui import progress
from tqdm import tqdm

ANNOTATION_PATH = 'data/annotations'
IMAGE_PATH = "data/images"
def download(url, path):
    if os.path.exists(path):
        return
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
            if chunk:
                f.write(chunk)
                f.flush()

def unzip(file, folder):
    with zipfile.ZipFile(file, 'r') as archive:
        # Extract all contents of the Zip file to a directory with the same name as the file
        archive.extractall(path=folder)
        # Print a message indicating that the extraction is complete
        print(f"Extracted contents from '{file}' to '{folder}' directory.")

def preprocess_coco_annotations(filename):
    ds = json.load(open(filename))
    images = pd.DataFrame(ds['images'])[['file_name', 'id']]
    annotations = pd.DataFrame(ds['annotations'])[['image_id', 'bbox', 'category_id']]
    merge_df = pd.merge(images, annotations, how='inner', left_on='id', right_on='image_id')
    def helper(g_df):
        f = list(zip(g_df['bbox'].tolist(), g_df['category_id'].tolist()))
        return f

    f_df = merge_df.groupby(['file_name', 'image_id']).apply(helper).reset_index()
    f_df = f_df.rename(columns={0:"box_category"})

    for _, row in tqdm(merge_df.iterrows(), total=merge_df.shape[0]):
        filename = row['file_name'].replace(".jpg",'.json')
        json.dump(row.to_dict(),open(f"{ANNOTATION_PATH}/{filename}",'w'))
    
    
def prepare_coco_dataset():
    images_url = "http://images.cocodataset.org/zips/train2017.zip"
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    prefix_path = "data/{file_name}.zip"
    
    download(annotations_url, prefix_path.format(file_name='annotation'))
    unzip(prefix_path.format(file_name='annotation'), ANNOTATION_PATH)
    preprocess_coco_annotations(f"{ANNOTATION_PATH}/annotations/instances_train2017.json")
    
    download(images_url, prefix_path.format(file_name='images'))
    unzip(prefix_path.format(file_name='images'), IMAGE_PATH)


if __name__ == "__main__":
    prepare_coco_dataset()