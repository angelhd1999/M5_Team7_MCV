from torch.utils.data import Dataset
import json
import random
import numpy as np
from scipy.io import loadmat

from utils import reduce_txt_embeds
from pathlib import Path
import os
from PIL import Image
from torchvision import  transforms

class COCOImagesAndCaptions(Dataset):
    SPLITS = ["train", "val", "test"]

    def __init__(
            self,
            dataset_path: str,
            split: str,
            task = None
    ):
        self.root_path = Path(dataset_path)
        self.task = task
        self.split = split
        self.tfms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        assert (self.root_path / "bert_feats.npy").exists(), "No textual features in data dir"
        assert (self.root_path / "resnet50_feats.mat").exists(), "No image features in data dir"
        assert split in self.SPLITS, "Invalid dataset split"
        assert (self.root_path / f"{self.split}.json").exists(), "No split data in data dir"

        # To determine partition files, use imgid from partition jsons
        with open(self.root_path / f"{self.split}.json", 'r') as f_json:
            split = json.load(f_json)
            indices = self._get_split_indices(split)
            self.images = self._get_split_images(split)

        self.text_features = np.load(str(self.root_path / "bert_feats.npy"), allow_pickle=True)
        self.text_features = self._mean_reduction(self.text_features)[indices]
        
        if self.task == None:
            self.img_features = loadmat(str(self.root_path / "resnet50_feats.mat"))['feats'].T[indices]
            

    def __getitem__(self, index):
        if self.task == None:
            images = self.img_features[index]  # (Images, FeatureSize)
        else:
            txt_features = self.text_features[index]  # (Images, FeatureSize)
            path = f'{self.root_path}/{self.split}'
            image = Image.open(os.path.join(path, self.images[index]))
            images = self.tfms(image)
        return txt_features, images

    def __len__(self):
        return self.images.shape[0]

    @staticmethod
    def _get_split_indices(json_obj):
        img_ids = []
        for ii in json_obj:
            img_ids.append(ii["imgid"])
        return np.asarray(img_ids)

    @staticmethod
    def _mean_reduction(embeds):
        aux1 = []
        for i in range(len(embeds)):
            aux2 = []
            for sent in embeds[i]:
                aux2.append(np.mean(sent, axis=0))
            aux1.append(aux2)
        return np.asarray(aux1)

    @staticmethod
    def _get_split_images(json_obj):
        img_ids = []
        for ii in json_obj:
            img_ids.append(ii["filename"])
        return np.asarray(img_ids)
    


class COCOImagesAndCaptionsBERT(Dataset):
    SPLITS = ["train", "val", "test"]

    def __init__(
            self,
            dataset_path: str,
            split: str
    ):
        root_path = Path(dataset_path)

        assert (Path("./results/task_d/bert_feats.npy")).exists(), "No textual features in data dir"
        assert (root_path / "resnet50_feats.mat").exists(), "No image features in data dir"
        assert split in self.SPLITS, "Invalid dataset split"
        assert (root_path / f"{split}.json").exists(), "No split data in data dir"

        # To determine partition files, use imgid from partition jsons
        with open(root_path / f"{split}.json", 'r') as f_json:
            split = json.load(f_json)
            indices = self._get_split_indices(split)
            

        self.img_features = loadmat(str(root_path / "resnet50_feats.mat"))['feats'].T[indices]
        self.text_features = np.load("./results/task_d/bert_feats.npy", allow_pickle=True)
        self.text_features = self._mean_reduction(self.text_features)[indices]
        
        

    def __getitem__(self, index):
        img_features = self.img_features[index]  # (Images, FeatureSize)
        txt_features = self.text_features[index]  # (Images, FeatureSize)
        
        return img_features, txt_features

    def __len__(self):
        return self.img_features.shape[0]

    @staticmethod
    def _get_split_indices(json_obj):
        img_ids = []
        for ii in json_obj:
            img_ids.append(ii["imgid"])
        return np.asarray(img_ids)



    @staticmethod
    def _mean_reduction(embeds):
        aux1 = []
        for i in range(len(embeds)):
            aux2 = []
            for sent in embeds[i]:
                aux2.append(np.mean(sent, axis=0))
            aux1.append(aux2)
        return np.asarray(aux1)