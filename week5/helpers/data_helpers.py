import fasttext
import os
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO

# Create dataset and dataloader
class COCOImageToTextDataset(torch.utils.data.Dataset):
    def __init__(self, coco, img_dir, transform):
        self.img_dir = img_dir
        self.coco = coco
        self.img_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]

        # ! I suspected following code provoked the next error when returning cat_ids_clean: RuntimeError: each element in list of batch should be of equal size
        coco = self.coco
        annotation_ids = coco.getAnnIds(img_id)
        annotations = coco.loadAnns(annotation_ids)

        # If the image has no annotations, return another image at random
        if len(annotations) == 0:
            # print(f'Img id {img_id} has no annotations. Returning another image at random.')
            while True:
                img_id = np.random.choice(self.img_ids)
                annotation_ids = coco.getAnnIds(img_id)
                annotations = coco.loadAnns(annotation_ids)
                if len(annotations) > 0:
                    break
            # print(f'Returning image id {img_id} instead.')

        positive_captions = [annotation['caption'] for annotation in annotations]
        # Get positive caption at random
        positive_caption = np.random.choice(positive_captions).replace("\n"," ") # ? Addded to avoid error: predict processes one line at a time (remove '\n')
        # Get negative caption at random
        while True:
            negative_img_id = np.random.choice(self.img_ids)
            if negative_img_id != img_id:
                break
        negative_annotation_ids = coco.getAnnIds(negative_img_id)
        negative_annotations = coco.loadAnns(negative_annotation_ids)
        negative_captions = [annotation['caption'] for annotation in negative_annotations]
        negative_caption = np.random.choice(negative_captions).replace("\n"," ") # ? Addded to avoid error: predict processes one line at a time (remove '\n')


        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        # Read as PIL image
        anchor_img = Image.open(img_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
        
        return anchor_img, positive_caption, negative_caption

# Create dataset and dataloader
class COCOTextToImageDataset(torch.utils.data.Dataset):
    def __init__(self, coco, img_dir, transform):
        self.img_dir = img_dir
        self.coco = coco
        self.img_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]

        # ! I suspected following code provoked the next error when returning cat_ids_clean: RuntimeError: each element in list of batch should be of equal size
        coco = self.coco
        annotation_ids = coco.getAnnIds(img_id)
        annotations = coco.loadAnns(annotation_ids)

        # If the image has no annotations, return another image at random
        if len(annotations) == 0:
            # print(f'Img id {img_id} has no annotations. Returning another image at random.')
            while True:
                img_id = np.random.choice(self.img_ids)
                annotation_ids = coco.getAnnIds(img_id)
                annotations = coco.loadAnns(annotation_ids)
                if len(annotations) > 0:
                    break
            # print(f'Returning image id {img_id} instead.')

        anchor_captions = [annotation['caption'] for annotation in annotations]
        # Get positive caption at random
        anchor_caption = np.random.choice(anchor_captions).replace("\n"," ") # ? Addded to avoid error: predict processes one line at a time (remove '\n')
        # Get negative caption at random
        while True:
            negative_img_id = np.random.choice(self.img_ids)
            if negative_img_id != img_id:
                break
        negative_img_info = self.coco.loadImgs([negative_img_id])[0]
        negative_img_path = os.path.join(self.img_dir, negative_img_info['file_name'])
        # Read as PIL image
        negative_img = Image.open(negative_img_path).convert('RGB')

        positive_img_info = self.coco.loadImgs([img_id])[0]
        positive_img_path = os.path.join(self.img_dir, positive_img_info['file_name'])
        # Read as PIL image
        positive_img = Image.open(positive_img_path).convert('RGB')

        if self.transform:
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return anchor_caption, positive_img, negative_img

# Load COCO dataset
def load_coco_dataset(anns_dir, data_dir, anns_key, img_dir_name):
    ann_file = os.path.join(anns_dir, 'captions_{}.json'.format(anns_key))
    img_dir = os.path.join(data_dir, img_dir_name)
    return COCO(ann_file), img_dir

# Create Dataloaders
def create_dataloaders(coco, img_dir, transform, batch_size, num_workers, mode):
    if mode == 'ITT':
        dataset = COCOImageToTextDataset(coco, img_dir, transform)
    elif mode == 'TTI':
        dataset = COCOTextToImageDataset(coco, img_dir, transform)
    else:
        raise ValueError('Mode must be either "ITT" or "TTI"')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader