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
import pickle
from sklearn.neighbors import NearestNeighbors
# import matplotlib.pyplot as plt
# Internal imports
from helpers.utils import parse_args
from helpers.data_helpers import load_coco_dataset, create_dataloaders, create_query_dataloaders
from helpers.model_helpers import TripletNetworkITT, TripletNetworkTTI, save_model
import random

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
LOAD_EMBS_PATH = args.load_embs_path
LOAD_EMBS_IDS_PATH = args.load_embs_ids_path
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

    args.ep_losses = []
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
        args.ep_losses.append(epoch_loss)  # Append the epoch_loss to the losses list
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        # If it's not the last epoch, save the model
        if epoch != NUM_EPOCHS - 1:
            save_model(model, MODE, TXT_EMB_MODEL, args, epoch, epoch_loss)
    # Save the model after training
    save_model(model, MODE, TXT_EMB_MODEL, args, epoch, epoch_loss, final=True)

def validate(model, criterion, mode, val_dataloader, device):
    model.eval()
    print('Starting validation')
    with torch.no_grad():

        IS_ITT = None
        if mode == 'ITT':
            IS_ITT = True
        elif mode == 'TTI':
            IS_ITT = False
        else:
            raise ValueError(f'Invalid mode: {mode}')

        args.ep_losses = []
        epoch_loss = 0.0
        num_batches = 0
        print(f'Starting Validation Epoch')
        for x, y, z in val_dataloader:
            # * Images need to be on the GPU if model is on the GPU
            # * Captions have to stay on the CPU if text embedding model is fasttext
            if IS_ITT:
                anchor_imgs = x.to(device)
                pos_captions = y
                neg_captions = z
            else:
                anchor_captions = x
                pos_imgs = y.to(device)
                neg_imgs = z.to(device)

            if IS_ITT:
                anchor_img_emb, pos_cap_embs, neg_cap_emb = model(anchor_imgs, pos_captions, neg_captions)
                loss = criterion(anchor_img_emb, pos_cap_embs, neg_cap_emb)
            else:
                anchor_cap_emb, pos_img_embs, neg_img_emb = model(anchor_captions, pos_imgs, neg_imgs)
                loss = criterion(anchor_cap_emb, pos_img_embs, neg_img_emb)

            epoch_loss += loss.item()
            num_batches += 1
            # Print the loss every 10 batches
            if num_batches % 10 == 0:
                print(f"Batch {num_batches}, Completed: {num_batches*64}/{len(val_dataloader.dataset)}, Loss: {loss.item():.4f}")

        epoch_loss /= num_batches
        args.ep_losses.append(epoch_loss)  # Append the epoch_loss to the losses list
        print(f"Validation Loss: {epoch_loss:.4f}")

def build_embeddings_db(model, img_dataloader, captions_dataloader, mode, txt_emb_model, device):
    model.eval()
    print('Building embeddings database')
    with torch.no_grad():
        IS_ITT = None
        if mode == 'ITT':
            IS_ITT = True
        elif mode == 'TTI':
            IS_ITT = False
        else:
            raise ValueError(f'Invalid mode: {mode}')

        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if IS_ITT:
            # Build the embeddings database for the captions
            it = 0
            cap_embs = []
            id_embs = []
            db_len = len(captions_dataloader.dataset)
            for captions, img_id in captions_dataloader:
                # * Captions have to stay on the CPU if text embedding model is fasttext
                caption_idx = 0
                # Captions is an array of tuples
                for caption in captions:
                    cap_emb = model(None, caption, None) # torch.Size([1, 256])
                    cap_embs.append(cap_emb)
                    id_embs.append((img_id, caption_idx))
                    caption_idx += 1
                if it % 100 == 0:
                    print(f'Processed {it} packs of captions of {db_len} for database building')
                    it += 1
            
            cap_embs = torch.stack(cap_embs).cpu().numpy()
            cap_embs = cap_embs.reshape(cap_embs.shape[0], -1)
            print('Saving database embeddings and image IDs to files')
            with open(f'cap_embs_{mode}_{txt_emb_model}_{datetime_str}', 'wb') as f:
                pickle.dump(cap_embs, f)
            with open(f'id_embs_{mode}_{txt_emb_model}_{datetime_str}', 'wb') as f:
                pickle.dump(id_embs, f)
            
            return cap_embs, id_embs
        
        else:
            # Build the embeddings database for the images
            it = 0
            img_embs = []
            id_embs = []
            db_len = len(img_dataloader.dataset)
            for img, img_id in img_dataloader:
                # * Images need to be on the GPU if model is on the GPU
                img_emb = model(None, img.to(device), None) # torch.Size([1, 256])
                img_embs.append(img_emb)
                id_embs.append(img_id)
                if it % 100 == 0:
                    print(f'Processed {it} packs of images of {db_len} for database building')
                    it += 1
            
            img_embs = torch.stack(img_embs).cpu().numpy()
            img_embs = img_embs.reshape(img_embs.shape[0], -1)
            print('Saving database embeddings and image IDs to files')
            with open(f'img_embs_{mode}_{txt_emb_model}_{datetime_str}', 'wb') as f:
                pickle.dump(img_embs, f)
            with open(f'id_embs_{mode}_{txt_emb_model}_{datetime_str}', 'wb') as f:
                pickle.dump(id_embs, f)
            
            return img_embs, id_embs

def unnormalize(img_tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for t, m, s in zip(img_tensor, mean, std):
        t.mul_(s).add_(m)
    return img_tensor

def perform_query(model, db_embs, id_embs, img_dataloader, captions_dataloader, device, mode, txt_emb_model):
    print('Building KNN')
    # Build the nearest neighbors model with the database embeddings
    nearest_neighbors = NearestNeighbors(n_neighbors=5)
    nearest_neighbors.fit(db_embs)
    print('KNN built')

    model.eval()
    print('Performing query')
    with torch.no_grad():
        IS_ITT = None
        if mode == 'ITT':
            IS_ITT = True
        elif mode == 'TTI':
            IS_ITT = False
        else:
            raise ValueError(f'Invalid mode: {mode}')
        
        if IS_ITT:
            # Perform the query for the images
            top1_acc = 0
            top5_acc = 0
            it = 0
            query_len = len(img_dataloader.dataset)
            for img, img_id in img_dataloader:
                if it % 100 == 0:
                    print(f'Processing query {it} of {query_len}')
                # * Images need to be on the GPU if model is on the GPU
                query_emb = model(img.to(device), None, None).cpu().numpy() # torch.Size([1, 256])
                query_emb = query_emb.reshape(1, -1)
                _, indices = nearest_neighbors.kneighbors(query_emb)
                
                query_img_id = img_id
                is_correct_top1 = False
                is_correct_top5 = False
                db_img_ids = []
                db_caption_idxs = []
                for ith, db_idx in enumerate(indices[0]):
                    db_img_id, db_caption_idx = id_embs[db_idx]
                    db_img_ids.append(db_img_id)
                    db_caption_idxs.append(db_caption_idx)
                    if query_img_id == db_img_id:
                        if ith == 0:
                            is_correct_top1 = True
                            is_correct_top5 = True
                        else:
                            is_correct_top5 = True

                # If its at top five or by chance 5% of the time
                if is_correct_top5 or random.random() < 0.05:
                    # Get query and db images
                    query_img, _ = img_dataloader.dataset[img_dataloader.dataset.img_ids.index(query_img_id)]
                    # Tensor to PIL image
                    query_img_unnorm = unnormalize(query_img.clone())
                    query_img_unnorm = transforms.ToPILImage()(query_img_unnorm)
                    
                    # Get query and db captions
                    query_caps, _ = captions_dataloader.dataset[captions_dataloader.dataset.img_ids.index(query_img_id)]
                    # Join the captions
                    query_caps = '\n'.join(query_caps)

                    db_caps_list = []
                    for db_caption_idx_r, db_img_id_r in zip(db_caption_idxs, db_img_ids):
                        db_caps, _ = captions_dataloader.dataset[captions_dataloader.dataset.img_ids.index(db_img_id_r)]
                        db_cap = db_caps[db_caption_idx_r]
                        db_caps_list.append(db_cap)
                    # Join the captions
                    db_caps_list = '\n'.join(db_caps_list)

                    if is_correct_top5:
                        # Save the query and db images
                        query_img_unnorm.save(f'{mode}_{txt_emb_model}_outputs/{it}_query.jpg')
                        # Save the query and db captions
                        with open(f'{mode}_{txt_emb_model}_outputs/{it}_gt.txt', 'w') as f:
                            f.write(query_caps)
                        with open(f'{mode}_{txt_emb_model}_outputs/{it}_retrieved.txt', 'w') as f:
                            f.write(db_caps_list)
                    else:
                        # Save the query and db images
                        query_img_unnorm.save(f'{mode}_{txt_emb_model}_outputs/{it}_query_wrong.jpg')
                        # Save the query and db captions
                        with open(f'{mode}_{txt_emb_model}_outputs/{it}_gt_wrong.txt', 'w') as f:
                            f.write(query_caps)
                        with open(f'{mode}_{txt_emb_model}_outputs/{it}_retrieved_wrong.txt', 'w') as f:
                            f.write(db_caps_list)
                if is_correct_top1:
                    top1_acc += 1
                if is_correct_top5:
                    top5_acc += 1
                it += 1
            print(f'Results for {mode} mode with {txt_emb_model} (DB size: {query_len})')
            print(f'Top1 accuracy: {top1_acc / query_len}')
            print(f'Top5 accuracy: {top5_acc / query_len}')
        else:
            # Perform the query for the captions
            top1_acc = 0
            top5_acc = 0
            it = 0
            query_len = len(captions_dataloader.dataset)
            for caps, img_id in captions_dataloader:
                if it % 100 == 0:
                    print(f'Processing query pack {it} of {query_len} (x5)')
                for cap_idx, cap in enumerate(caps):
                    # * Images need to be on the GPU if model is on the GPU
                    query_emb = model(cap, None, None).cpu().numpy()
                    query_emb = query_emb.reshape(1, -1)
                    _, indices = nearest_neighbors.kneighbors(query_emb)

                    query_img_id = img_id
                    query_caption_idx = cap_idx
                    is_correct_top1 = False
                    is_correct_top5 = False
                    db_img_ids = []
                    for ith, db_idx in enumerate(indices[0]):
                        db_img_id = id_embs[db_idx]
                        db_img_ids.append(db_img_id)
                        if query_img_id == db_img_id:
                            if ith == 0:
                                is_correct_top1 = True
                                is_correct_top5 = True
                            else:
                                is_correct_top5 = True
                    
                    # If its at top five or by chance 5% of the time
                    if is_correct_top5 or random.random() < 0.05:
                        # Get query and db images
                        query_img, _ = img_dataloader.dataset[img_dataloader.dataset.img_ids.index(query_img_id)]
                        # Tensor to PIL image
                        query_img_unnorm = unnormalize(query_img.clone())
                        query_img_unnorm = transforms.ToPILImage()(query_img_unnorm)

                        db_imgs_r = []
                        for db_img_id_r in db_img_ids:
                            db_img_r, _ = img_dataloader.dataset[captions_dataloader.dataset.img_ids.index(db_img_id_r)]
                            db_imgs_r.append(db_img_r)

                        if is_correct_top5:
                            # Save the query and db images
                            query_img_unnorm.save(f'{mode}_{txt_emb_model}_outputs/{it}_query.jpg')
                            for i, db_img_r in enumerate(db_imgs_r):
                                db_img_unnorm_r = unnormalize(db_img_r.clone())
                                db_img_unnorm_r = transforms.ToPILImage()(db_img_unnorm_r)
                                db_img_unnorm_r.save(f'{mode}_{txt_emb_model}_outputs/{it}_retrieved_{i}.jpg')
                            # Save the query and db captions
                            with open(f'{mode}_{txt_emb_model}_outputs/{it}_query.txt', 'w') as f:
                                f.write(cap)
                        else:
                            # Save the query and db images
                            query_img_unnorm.save(f'{mode}_{txt_emb_model}_outputs/{it}_query_wrong.jpg')
                            for i, db_img_r in enumerate(db_imgs_r):
                                db_img_unnorm_r = unnormalize(db_img_r.clone())
                                db_img_unnorm_r = transforms.ToPILImage()(db_img_unnorm_r)
                                db_img_unnorm_r.save(f'{mode}_{txt_emb_model}_outputs/{it}_retrieved_wrong_{i}.jpg')
                            # Save the query and db captions
                            with open(f'{mode}_{txt_emb_model}_outputs/{it}_query_wrong.txt', 'w') as f:
                                f.write(cap)
                        it += 1
                    if is_correct_top1:
                        top1_acc += 1
                    if is_correct_top5:
                        top5_acc += 1  
            print(f'Results for {mode} mode with {txt_emb_model} (DB size: {query_len})')
            print(f'Top1 accuracy: {top1_acc / query_len}')
            print(f'Top5 accuracy: {top5_acc / query_len}')

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
        coco_test, test_img_dir = load_coco_dataset(VAL_TEST_ANNS_DIR, DATA_DIR, 'miniquery2014', 'val2014')
        img_dataloader, captions_dataloader = create_query_dataloaders(coco_test, test_img_dir, transform)

    # Create the triplet network and set up the training process
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ? OLD
    # Instantiate the custom model with the desired embedding dimension
    # backbone_body = fasterrcnn_resnet50_fpn(weights='DEFAULT').backbone
    # feature_extractor = CustomModel(backbone_body, EMBEDDING_DIM)
    if not TEST: # ! Test is expected to be done alone
        if MODE == 'ITT':
            model = TripletNetworkITT(TXT_EMB_MODEL, EMBEDDING_DIM, device).to(device)
            print('ITT model created')
        elif MODE == 'TTI':
            model = TripletNetworkTTI(TXT_EMB_MODEL, EMBEDDING_DIM, device).to(device)
            print('TTI model created')
        # ? Maybe use the TripletMarginLoss from the pytorch_metric_learning
        criterion = nn.TripletMarginLoss(margin=MARGIN)

        trained = False
        if TRAIN:
            train(model, criterion, MODE, train_dataloader, NUM_EPOCHS, SCHEDULER, device)
            end = time.time()
            print(f"Training time: {end - start} seconds")
        
        if VALIDATE:
            if not trained:
                # Load the model weights
                model.load_state_dict(torch.load(LOAD_MODEL_PATH))
            validate(model, criterion, MODE, val_dataloader, device)
        
    if TEST:
        if MODE == 'ITT':
            model = TripletNetworkITT(TXT_EMB_MODEL, EMBEDDING_DIM, device, test = True).to(device)
            print('ITT model created')
        elif MODE == 'TTI':
            model = TripletNetworkTTI(TXT_EMB_MODEL, EMBEDDING_DIM, device, test = True).to(device)
            print('TTI model created')
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))
        if os.path.exists(LOAD_EMBS_PATH) and os.path.exists(LOAD_EMBS_IDS_PATH):
            print('Loading embeddings and ids from file')
            with open(LOAD_EMBS_PATH, 'rb') as f:
                db_embs = pickle.load(f)
            with open(LOAD_EMBS_IDS_PATH, 'rb') as f:
                id_embs = pickle.load(f)
        else:
            db_embs, id_embs = build_embeddings_db(model, img_dataloader, captions_dataloader, MODE, TXT_EMB_MODEL, device)
        perform_query(model, db_embs, id_embs, img_dataloader, captions_dataloader, device, MODE, TXT_EMB_MODEL)

if __name__ == '__main__':
    main()