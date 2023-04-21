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
# Internal imports
from helpers.utils import parse_args
from helpers.data_helpers import load_coco_dataset, create_dataloaders
from helpers.model_helpers import CustomModel, TripletNetwork, save_model

args = parse_args()
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
    train_dataloader = create_dataloaders(coco_train, train_img_dir, transform, BATCH_SIZE, NUM_WORKERS)
if VALIDATE:
    val_dataloader = create_dataloaders(coco_val, val_img_dir, transform, BATCH_SIZE, NUM_WORKERS)

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