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
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pickle

# PARAMS
TRAIN = False
VALIDATE = False
ONLY_TEST = True
LOAD_MODEL_PATH = 'triplet_network_base_task_e_lr_0_0001.pth'

start = time.time()

# Load COCO dataset
def load_coco_dataset(data_dir, mode='train2014'):
    ann_file = os.path.join(data_dir, 'instances_{}.json'.format(mode))
    img_dir = os.path.join(data_dir, mode)
    return COCO(ann_file), img_dir

data_dir = '../../../mcv/datasets/COCO/'
coco_train, train_img_dir = load_coco_dataset(data_dir, mode='train2014')
coco_val, val_img_dir = load_coco_dataset(data_dir, mode='val2014')

# Load the JSON annotations file
with open('../../../mcv/datasets/COCO/mcv_image_retrieval_annotations.json', 'r') as f:
    annotations = json.load(f)

train_ann = annotations['train']
val_ann = annotations['val']
query_ann = annotations['test']
database_ann = annotations['database']

# * Added the GeneralizedRCNNTransform to the transforms
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    # transforms.Resize(size=(800,), max_size=1333 ), # * Added from GeneralizedRCNNTransform (originally: transforms.Resize(min_size=(800,), max_size=1333, mode='bilinear'))
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # * Also in GeneralizedRCNNTransform
])

# Create dataset and dataloader
class COCOImageRetrievalDataset(torch.utils.data.Dataset):
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
    
# Create a function to get the category ids for a list of image ids
def get_category_ids(coco, img_id):
    annotation_ids = coco.getAnnIds(img_id)
    annotations = coco.loadAnns(annotation_ids)
    cat_ids = []
    for i in range(len(annotations)):
        entity_id = annotations[i]["category_id"]
        cat_ids.append(entity_id)
    return set(cat_ids), cat_ids # Remove duplicates

def create_dataloaders(coco, img_dir, transform, batch_size=64, num_workers=4):
    dataset = COCOImageRetrievalDataset(coco, img_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

if not ONLY_TEST:
    train_dataloader = create_dataloaders(coco_train, train_img_dir, transform)
    val_dataloader = create_dataloaders(coco_val, val_img_dir, transform)

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
        if positive is None: # Added for evaluation
            return anchor_embedding
        positive_embedding = self.feature_extractor(positive)
        negative_embedding = self.feature_extractor(negative)
        return anchor_embedding, positive_embedding, negative_embedding

# Use the TripletMarginLoss from the pytorch_metric_learning
criterion = nn.TripletMarginLoss(margin=1.0)

# Instantiate the custom model with the desired embedding dimension
backbone_body = fasterrcnn_resnet50_fpn(weights='DEFAULT').backbone
embedding_dim = 64
feature_extractor = CustomModel(backbone_body, embedding_dim)

# * NEW: Freezing all layers except the last one
# # Freeze the parameters of the feature extractor
# for param in feature_extractor.parameters():
#     param.requires_grad = False

# # Set requires_grad to True only for the parameters of the fc layer
# for param in feature_extractor.fc.parameters():
#     param.requires_grad = True

# Create the triplet network and set up the training process
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TripletNetwork(feature_extractor).to(device)

if TRAIN:
    # !OLD: 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # # * New: Create a new optimizer that only updates the parameters of the fc layer
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    # * New: Create a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for anchor_imgs, anchor_ids in train_dataloader:
            anchor_imgs = anchor_imgs.to(device)
            # print('ATTENTION')
            # print(anchor_ids)
            # Transform the anchor_ids to a list
            anchor_ids = anchor_ids.tolist()

            positive_imgs = []
            negative_imgs = []
            for anchor_id in anchor_ids:
                set_anchor_cat_ids, list_anchor_cat_ids = get_category_ids(coco_train, anchor_id)
                # Select a positive image
                shared_cat_id = np.random.choice(list_anchor_cat_ids)
                positive_img_id = np.random.choice(train_ann[str(shared_cat_id)])
                positive_img, _ = train_dataloader.dataset[train_dataloader.dataset.img_ids.index(positive_img_id)]

                # Select a negative image
                while True:
                    negative_img_id = np.random.choice(train_dataloader.dataset.img_ids)
                    negative_img, negative_cat_id = train_dataloader.dataset[train_dataloader.dataset.img_ids.index(negative_img_id)]
                    set_negative_cat_ids, _ = get_category_ids(coco_train, negative_cat_id)
                    if not set_anchor_cat_ids.intersection(set_negative_cat_ids):
                        break
                
                positive_imgs.append(positive_img)
                negative_imgs.append(negative_img)

            # Stack the images
            positive_imgs = torch.stack(positive_imgs).to(device)
            negative_imgs = torch.stack(negative_imgs).to(device)
            optimizer.zero_grad()

            # TODO: Next task, check if the model is correctly freezed (only the last layer should be trainable)
            anchor_embeddings, positive_embeddings, negative_embeddings = model(anchor_imgs, positive_imgs, negative_imgs)
            # print('Anchor embeddings')
            # # print(anchor_embeddings)
            # print(anchor_embeddings.shape)
            # print('Positive embeddings')
            # print(positive_embeddings.shape)
            # print('Negative embeddings')
            # print(negative_embeddings.shape)

            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            # Print the loss every 10 batches
            if num_batches % 10 == 0:
                print(f"Batch {num_batches}, Completed: {num_batches*64}/{len(train_dataloader.dataset)}, Loss: {loss.item():.4f}")
            # Update the learning rate
            scheduler.step()

        epoch_loss /= num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    # # Get date string
    # current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # # Saving the model weights
    # print(f'Saving the model weights as triplet_network_{current_date}.pth')
    torch.save(model.state_dict(), f'triplet_network_base_task_e_nointended.pth')
    end = time.time()
    print(f"Training time: {end - start} seconds")
else:
    # Load the model weights
    model.load_state_dict(torch.load(LOAD_MODEL_PATH))

if VALIDATE:
    val_epoch_loss = 0.0
    num_batches = 0
    print('Validating the model')
    # Validate the model
    model.eval()
    with torch.no_grad():
        # ! Currently wrong: It should iterate only on the validation defined by them at mcv_annotations...
        for anchor_imgs, anchor_ids in val_dataloader:
            anchor_imgs = anchor_imgs.to(device)
            # Transform the anchor_ids to a list
            anchor_ids = anchor_ids.tolist()

            positive_imgs = []
            negative_imgs = []
            for anchor_id in anchor_ids:
                set_anchor_cat_ids, list_anchor_cat_ids = get_category_ids(coco_val, anchor_id)
                # Select a positive image
                shared_cat_id = np.random.choice(list_anchor_cat_ids)
                positive_img_id = np.random.choice(val_ann[str(shared_cat_id)])
                positive_img, _ = val_dataloader.dataset[val_dataloader.dataset.img_ids.index(positive_img_id)]

                # Select a negative image
                while True:
                    negative_img_id = np.random.choice(val_dataloader.dataset.img_ids)
                    negative_img, negative_cat_id = val_dataloader.dataset[val_dataloader.dataset.img_ids.index(negative_img_id)]
                    set_negative_cat_ids, _ = get_category_ids(coco_val, negative_cat_id)
                    if not set_anchor_cat_ids.intersection(set_negative_cat_ids):
                        break
                
                positive_imgs.append(positive_img)
                negative_imgs.append(negative_img)

            # Stack the images
            positive_imgs = torch.stack(positive_imgs).to(device)
            negative_imgs = torch.stack(negative_imgs).to(device)
            
            anchor_embeddings, positive_embeddings, negative_embeddings = model(anchor_imgs, positive_imgs, negative_imgs)

            val_loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            val_epoch_loss += val_loss.item()
            num_batches += 1
            # Print the loss every 10 batches
            if num_batches % 10 == 0:
                print(f"Batch {num_batches}, Val. Loss: {val_loss.item():.4f}")

    val_epoch_loss /= num_batches
    print(f"Val. Loss: {val_epoch_loss:.4f}")

# Evaluation zone
# Read all_image_ids_classes.json
with open('all_image_ids_classes.json', 'r') as f:
    all_image_ids_classes = json.load(f)

# Get them for train, val, database and query
train_image_ids_classes = all_image_ids_classes['train']
val_image_ids_classes = all_image_ids_classes['val']
database_image_ids_classes = all_image_ids_classes['database']
query_image_ids_classes = all_image_ids_classes['test']

# Evaluate the model using the evaluation functions
database_dataloader = create_dataloaders(coco_train, train_img_dir, transform, batch_size=1)
query_dataloader = create_dataloaders(coco_val, val_img_dir, transform, batch_size=1)

def unnormalize(img_tensor, mean, std):
    for t, m, s in zip(img_tensor, mean, std):
        t.mul_(s).add_(m)
    return img_tensor

def plot_retrieval_results(query_img, retrieved_img, query_img_classes, query_img_classes_original, retrieved_img_classes, retrieved_img_classes_original, is_correct, query_img_id, retrieved_img_id):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    query_img_unnorm = unnormalize(query_img.clone(), mean, std)
    retrieved_img_unnorm = unnormalize(retrieved_img.clone(), mean, std)

    retrieval_imgs_path = 'retrieval_imgs'
    
    # apply coco_id_class_mapping to these arrays 
    query_img_classes = [coco_id_class_mapping[int(c)] for c in query_img_classes]
    retrieved_img_classes = [coco_id_class_mapping[int(c)] for c in retrieved_img_classes]
    query_img_classes_original = [coco_id_class_mapping[int(c)] for c in query_img_classes_original]
    retrieved_img_classes_original = [coco_id_class_mapping[int(c)] for c in retrieved_img_classes_original]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Correct retrieval' if is_correct else 'Incorrect retrieval', fontsize=16)

    axs[0].imshow(query_img_unnorm.permute(1, 2, 0).numpy())
    axs[0].set_title(
    f'Query Image ID: {query_img_id}\n\
    Target Classes: {", ".join(query_img_classes)}\n\
    \n\
    Original Classes:\n\
    {", ".join(query_img_classes_original)}'
    )
    axs[0].axis('off')

    axs[1].imshow(retrieved_img_unnorm.permute(1, 2, 0).numpy())
    axs[1].set_title(
    f'Retrieved Image ID: {retrieved_img_id}\n\
    Target Classes: {", ".join(retrieved_img_classes)}\n\
    \n\
    Original Classes:\n\
    {", ".join(retrieved_img_classes_original)}'
    )
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig(f"{retrieval_imgs_path}/extra_{'correct' if is_correct else 'incorrect'}_retrieval_{query_img_id}_{retrieved_img_id}.png")
    plt.close(fig)


def evaluate_image_retrieval_NN(query_loader, database_loader, coco_val, model, query_image_ids_classes, database_image_ids_classes, val_image_ids_classes, train_image_ids_classes, database_embeddings_file='database_embeddings.pkl', database_img_ids_file='database_img_ids.pkl'):
    correct = 0
    correct_original = 0
    top_k = 1
    model.eval()
    plot_percentage = 0.1  # Adjust this value to control the percentage of results you want to save as plots
    db_len = len(list(database_image_ids_classes.keys()))
    query_len = len(list(query_image_ids_classes.keys()))

    with torch.no_grad():
        # Check if the database embeddings and image IDs are already saved
        if os.path.exists(database_embeddings_file) and os.path.exists(database_img_ids_file):
            with open(database_embeddings_file, 'rb') as f:
                database_embeddings = pickle.load(f)
            with open(database_img_ids_file, 'rb') as f:
                database_img_ids = pickle.load(f)
            print('Loaded database embeddings and image IDs from files')
        else:
            # Build the database embeddings
            database_embeddings = []
            database_img_ids = []
            it = 0

            for database_img_id, _ in database_image_ids_classes.items():
                database_img_id = int(database_img_id)
                database_img_ids.append(database_img_id)
                database_img, _ = database_loader.dataset[database_loader.dataset.img_ids.index(database_img_id)]
                database_img_embedding = model.feature_extractor(database_img.to(device))
                database_embeddings.append(database_img_embedding)

                if it % 100 == 0:
                    print(f'Processed {it} images of {db_len} for database building')
                it += 1

            database_embeddings = torch.stack(database_embeddings).cpu().numpy()
            database_embeddings = database_embeddings.reshape(database_embeddings.shape[0], -1)

            print('Saving database embeddings and image IDs to files')
            with open(database_embeddings_file, 'wb') as f:
                pickle.dump(database_embeddings, f)
            with open(database_img_ids_file, 'wb') as f:
                pickle.dump(database_img_ids, f)

        # Build the nearest neighbors model with the database embeddings
        nearest_neighbors = NearestNeighbors(n_neighbors=top_k)
        nearest_neighbors.fit(database_embeddings)

        # Iterate over the query images
        for query_img_id, query_img_info in query_image_ids_classes.items():
            query_img_id = int(query_img_id)
            query_img, _ = query_loader.dataset[query_loader.dataset.img_ids.index(query_img_id)]
            query_img_embedding = model.feature_extractor(query_img.to(device)).cpu().numpy()
            query_img_embedding = query_img_embedding.reshape(1, -1)
            query_img_classes = query_img_info['classes']
            # query_img_classes_original = val_image_ids_classes[str(query_img_id)]['classes']
            # Retrieve all the original classes from coco_val
            query_img_classes_original, _ = list(get_category_ids(coco_val, query_img_id))
            # Map values to strings
            query_img_classes_original = [str(c) for c in query_img_classes_original]
            # print(f"Query original classes: {query_img_classes_original}")

            # Find the nearest neighbors for the query image
            _, indices = nearest_neighbors.kneighbors(query_img_embedding)
            
            # Check if the query image shares classes with the nearest neighbors
            is_correct = False
            # Check if the query image shares classes with the nearest neighbors
            for idx in indices[0]:
                database_img_id = database_img_ids[idx]
                database_img_info = database_image_ids_classes[str(database_img_id)]
                database_img_classes = database_img_info['classes']
                database_img_classes_original = train_image_ids_classes[str(database_img_id)]['classes']

                if set(query_img_classes_original).intersection(set(database_img_classes_original)):
                    correct_original += 1

                if set(query_img_classes).intersection(set(database_img_classes)):
                    print(f"{query_img_id} shares classes with the {top_k}: {database_img_id}")
                    print(f"Query classes: {query_img_classes}")
                    print(f"Database classes: {database_img_classes}")
                    correct += 1
                    is_correct = True
                    break

            # if random.random() < plot_percentage:
            database_img, _ = database_loader.dataset[database_loader.dataset.img_ids.index(database_img_id)]
            plot_retrieval_results(query_img.cpu(), database_img.cpu(), query_img_classes, query_img_classes_original, database_img_classes, database_img_classes_original, is_correct, query_img_id, database_img_id)

    return correct / query_len, correct_original / query_len



print("Evaluating the model...")
retrieval_accuracy, retrieval_accuracy_original = evaluate_image_retrieval_NN(query_dataloader, database_dataloader, coco_val, model, query_image_ids_classes, database_image_ids_classes, val_image_ids_classes, train_image_ids_classes)
print(f"Image retrieval accuracy percentage: {retrieval_accuracy * 100:.2f}%")
print(f"Image retrieval accuracy percentage (original classes): {retrieval_accuracy_original * 100:.2f}%")