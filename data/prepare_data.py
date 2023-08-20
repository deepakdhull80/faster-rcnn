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
from clint.textui import progress

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
    annotation = json.load(open(filename))
    

def prepare_coco_dataset():
    images_url = "http://images.cocodataset.org/zips/train2017.zip"
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    prefix_path = "data/{file_name}.zip"
    
    download(annotations_url, prefix_path.format(file_name='annotation'))
    unzip(prefix_path.format(file_name='annotation'), ANNOTATION_PATH)
    
    # download(images_url, prefix_path.format(file_name='images'))

if __name__ == "__main__":
    prepare_coco_dataset()