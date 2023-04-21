import argparse
import fasttext
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
import json
from PIL import Image
import time
import datetime
# from sklearn.neighbors import NearestNeighbors
# import matplotlib.pyplot as plt
# import pickle

# Argument parser
parser = argparse.ArgumentParser(description='Image-to-text model training.')
# *Static variables
parser.add_argument('--data_dir', type=str, default='../../../mcv/datasets/COCO/', help='Path to the data directory.')
parser.add_argument('--val_test_anns_dir', type=str, default='./cocosplit', help='Path to the validation/test annotations directory.')
## Variable variables
# *Execution variables
parser.add_argument('--train', action='store_true', help='Train the model.')
parser.add_argument('--validate', action='store_true', help='Validate the model.')
parser.add_argument('--test', action='store_true', help='Test the model.')
parser.add_argument('--mode', type=str, options=['ITT', 'TTI'], default='ITT', help='Mode to run the model.')
parser.add_argument('--txt_emb', type=str, options=['fasttext', 'bert'], default='fasttext', help='Text embedding to use.')
parser.add_argument('--load_model_path', type=str, default='triplet_network_base_task_e_lr_0_0001.pth', help='Path to load the model.')
parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs.')
parser.add_argument('--scheduler', action='store_true', help='Use scheduler.')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers.')
# *Model variables
parser.add_argument('--img_size', type=int, default=224, help='Image size.')
parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate.')
parser.add_argument('--step_size', type=int, default=10, help='Step size for the learning rate scheduler.')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for the learning rate scheduler.')
parser.add_argument('--margin', type=float, default=1.0, help='Margin for the TripletLoss.')
args = parser.parse_args()

# *Static variables
DATA_DIR = args.data_dir
VAL_TEST_ANNS_DIR = args.val_test_anns_dir
## Variable variables
# *Execution variables
TRAIN = args.train
VALIDATE = args.validate
TEST = args.test
MODE = args.mode
TXT_EMB = args.txt_emb
LOAD_MODEL_PATH = args.load_model_path
NUM_EPOCHS = args.num_epochs
SCHEDULER = args.scheduler
NUM_WORKERS = args.num_workers
# *Model variables
IMG_SIZE = args.img_size
EMBEDDING_DIM = args.embedding_dim
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
STEP_SIZE = args.step_size
GAMMA = args.gamma
MARGIN = args.margin

start = time.time()

# Load COCO dataset
def load_coco_dataset(anns_dir, data_dir, anns_key, img_dir_name):
    ann_file = os.path.join(anns_dir, 'captions_{}.json'.format(anns_key))
    img_dir = os.path.join(data_dir, img_dir_name)
    return COCO(ann_file), img_dir

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

        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        # Read as PIL image
        img = Image.open(img_path).convert('RGB')
    
        if self.transform:
            img = self.transform(img)
        
        return img, img_id

def create_dataloaders(coco, img_dir, transform, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    dataset = COCOImageToTextDataset(coco, img_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

class CustomModel(nn.Module):
    def __init__(self, backbone_body, embedding_dim):
        super(CustomModel, self).__init__()
        self.backbone_body = backbone_body
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Add this line
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.backbone_body(x)
        # Access the last layer output
        x = x["pool"]
        x = self.avgpool(x)  # Add this line
        x = torch.flatten(x, 1)  # Add this line to flatten the tensor
        x = self.fc(x)
        return x

# Implement the triplet network model
class TripletNetwork(nn.Module):
    def __init__(self, feature_extractor):
        super(TripletNetwork, self).__init__()
        self.feature_extractor = feature_extractor

    def forward(self, anchor, positive, negative):
        anchor_embedding = self.feature_extractor(anchor)
        # if positive is None: # Added for evaluation
        #     return anchor_embedding
        positive_embedding = self.feature_extractor(positive)
        negative_embedding = self.feature_extractor(negative)
        return anchor_embedding, positive_embedding, negative_embedding

# Save model and args
def save_model(model, mode, txt_emb, args, epoch, loss, final=False):
    # Get date string
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Savename
    if final:
        save_name = f'model{mode}_{txt_emb}_final_{current_date}.pth'
    else:
        save_name = f'model{mode}_{txt_emb}_{epoch}_{current_date}.pth'
    # Saving the model weights
    print(f'Saving the model weights as {save_name}')
    torch.save(model.state_dict(), f'{save_name}.pth')
    # Add loss to args
    args.loss = loss
    # Save args
    with open(f'{save_name}.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

coco_train, train_img_dir = load_coco_dataset(DATA_DIR, DATA_DIR, 'train2014', 'train2014')
coco_val, val_img_dir = load_coco_dataset(VAL_TEST_ANNS_DIR, DATA_DIR, 'validation2014', 'val2014')
coco_test, test_img_dir = load_coco_dataset(VAL_TEST_ANNS_DIR, DATA_DIR, 'test2014', 'val2014')

# * Added the GeneralizedRCNNTransform to the transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # * Also in GeneralizedRCNNTransform
])

if TRAIN:
    train_dataloader = create_dataloaders(coco_train, train_img_dir, transform)
if VALIDATE:
    val_dataloader = create_dataloaders(coco_val, val_img_dir, transform)

# Create the triplet network and set up the training process
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use the TripletMarginLoss from the pytorch_metric_learning
criterion = nn.TripletMarginLoss(margin=MARGIN)

# Instantiate the custom model with the desired embedding dimension
backbone_body = fasterrcnn_resnet50_fpn(weights='DEFAULT').backbone
feature_extractor = CustomModel(backbone_body, EMBEDDING_DIM)

model = TripletNetwork(feature_extractor).to(device)

if TRAIN:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if SCHEDULER:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for anchor_imgs, anchor_ids in train_dataloader:
            anchor_imgs = anchor_imgs.to(device)
            # Transform the anchor_ids to a list
            anchor_ids = anchor_ids.tolist()

            positive_imgs = []
            negative_imgs = []
            for anchor_id in anchor_ids:
                # Get the positive image
                print(f'Anchor id: {anchor_id}')
                # positive_imgs.append(positive_img)
                # negative_imgs.append(negative_img)

            # Stack the images
            positive_imgs = torch.stack(positive_imgs).to(device)
            negative_imgs = torch.stack(negative_imgs).to(device)
            optimizer.zero_grad()

            anchor_embeddings, positive_embeddings, negative_embeddings = model(anchor_imgs, positive_imgs, negative_imgs)

            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            if SCHEDULER:
                # Update the learning rate
                scheduler.step()
            # Print the loss every 10 batches
            if num_batches % 10 == 0:
                print(f"Batch {num_batches}, Completed: {num_batches*64}/{len(train_dataloader.dataset)}, Loss: {loss.item():.4f}")

        epoch_loss /= num_batches
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")
        # If it's not the last epoch, save the model
        if epoch != NUM_EPOCHS - 1:
            save_model(model, MODE, TXT_EMB, args, epoch, epoch_loss)
    
    # Save the model after training
    save_model(model, MODE, TXT_EMB, args, epoch, epoch_loss, final=True)
    end = time.time()
    print(f"Training time: {end - start} seconds")
else:
    # Load the model weights
    model.load_state_dict(torch.load(LOAD_MODEL_PATH))