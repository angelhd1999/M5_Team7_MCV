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
        # cat_ids = []
        # for i in range(len(annotations)):
        #     entity_id = annotations[i]["category_id"]
        #     cat_ids.append(entity_id)
        # cat_ids_clean = list(set(cat_ids))

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

        for annotation in annotations:
            print(annotation['caption'])

        

        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        # Read as PIL image
        img = Image.open(img_path).convert('RGB')

        # save image
        img.save('test.png')
        exit()

        if self.transform:
            img = self.transform(img)
        
        return img, img_id

# Load COCO dataset
def load_coco_dataset(anns_dir, data_dir, anns_key, img_dir_name):
    ann_file = os.path.join(anns_dir, 'captions_{}.json'.format(anns_key))
    img_dir = os.path.join(data_dir, img_dir_name)
    return COCO(ann_file), img_dir

# Create Dataloaders
def create_dataloaders(coco, img_dir, transform, batch_size, num_workers):
    dataset = COCOImageToTextDataset(coco, img_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader