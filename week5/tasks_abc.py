import argparse
import fasttext
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
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
from helpers.model_helpers import CustomModel, TripletNetworkITT, save_model

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
TXT_EMB_MODEL = args.txt_emb_model
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

def train(model, criterion, mode, train_dataloader, num_epochs, scheduler, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if SCHEDULER:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    IS_ITT = None
    if mode == 'ITT':
        IS_ITT = True
    elif mode == 'TTI':
        IS_ITT = False
    else:
        raise ValueError(f'Invalid mode: {mode}')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        print(f'Starting Epoch {epoch + 1}/{num_epochs}')
        for x, y, z in train_dataloader:
            # * Images need to be on the GPU if model is on the GPU
            # * Captions have to stay on the CPU if text embedding model is fasttext
            if IS_ITT:
                anchor_imgs = x.to(device)
                pos_captions = y
                neg_captions = z
                print('pos_captions', pos_captions)
            else:
                anchor_captions = x
                pos_imgs = y.to(device)
                neg_imgs = z.to(device)

            optimizer.zero_grad()

            if IS_ITT:
                anchor_img_emb, pos_cap_embs, neg_cap_emb = model(anchor_imgs, pos_captions, neg_captions)
                loss = criterion(anchor_img_emb, pos_cap_embs, neg_cap_emb)
            else:
                anchor_cap_emb, pos_img_embs, neg_img_emb = model(anchor_captions, pos_imgs, neg_imgs)
                loss = criterion(anchor_cap_emb, pos_img_embs, neg_img_emb)

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
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        # If it's not the last epoch, save the model
        if epoch != NUM_EPOCHS - 1:
            save_model(model, MODE, TXT_EMB_MODEL, args, epoch, epoch_loss)
    # Save the model after training
    save_model(model, MODE, TXT_EMB_MODEL, args, epoch, epoch_loss, final=True)

def main():
    start = time.time()

    # * Added the GeneralizedRCNNTransform to the transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # * Also in GeneralizedRCNNTransform
    ])

    if TRAIN:
        print('Train enabled')
        coco_train, train_img_dir = load_coco_dataset(DATA_DIR, DATA_DIR, 'train2014', 'train2014')
        train_dataloader = create_dataloaders(coco_train, train_img_dir, transform, BATCH_SIZE, NUM_WORKERS, MODE)
    if VALIDATE:
        print('Validation enabled')
        coco_val, val_img_dir = load_coco_dataset(VAL_TEST_ANNS_DIR, DATA_DIR, 'validation2014', 'val2014')
        val_dataloader = create_dataloaders(coco_val, val_img_dir, transform, BATCH_SIZE, NUM_WORKERS, MODE)
    if TEST:
        print('Test enabled')
        coco_test, test_img_dir = load_coco_dataset(VAL_TEST_ANNS_DIR, DATA_DIR, 'test2014', 'val2014')
        test_dataloader = create_dataloaders(coco_test, test_img_dir, transform, BATCH_SIZE, NUM_WORKERS, MODE)

    # Create the triplet network and set up the training process
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ? OLD
    # Instantiate the custom model with the desired embedding dimension
    # backbone_body = fasterrcnn_resnet50_fpn(weights='DEFAULT').backbone
    # feature_extractor = CustomModel(backbone_body, EMBEDDING_DIM)
    
    if MODE == 'ITT':
        model = TripletNetworkITT(TXT_EMB_MODEL, EMBEDDING_DIM).to(device)
        print('ITT model created')
    elif MODE == 'TTI':
        raise NotImplementedError('TripletNetworkTTI not implemented yet')
    # ? Maybe use the TripletMarginLoss from the pytorch_metric_learning
    criterion = nn.TripletMarginLoss(margin=MARGIN)

    if TRAIN:
        train(model, criterion, MODE, train_dataloader, NUM_EPOCHS, SCHEDULER, device)
        end = time.time()
        print(f"Training time: {end - start} seconds")
    else:
        # Load the model weights
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))

if __name__ == '__main__':
    main()