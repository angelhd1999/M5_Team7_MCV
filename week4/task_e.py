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
from pytorch_metric_learning import losses

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
test_ann = annotations['test']
database_ann = annotations['database']

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        # coco = self.coco
        # annotation_ids = coco.getAnnIds(img_id)
        # annotations = coco.loadAnns(annotation_ids)
        # cat_ids = []
        # for i in range(len(annotations)):
        #     entity_id = annotations[i]["category_id"]
        #     cat_ids.append(entity_id)
        # cat_ids_clean = list(set(cat_ids))

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

def create_dataloaders(coco, img_dir, transform, batch_size=16, num_workers=1):
    dataset = COCOImageRetrievalDataset(coco, img_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

train_dataloader = create_dataloaders(coco_train, train_img_dir, transform)
val_dataloader = create_dataloaders(coco_val, val_img_dir, transform)

# Implement the triplet network model
class TripletNetwork(nn.Module):
    def __init__(self, feature_extractor):
        super(TripletNetwork, self).__init__()
        self.feature_extractor = feature_extractor

    def forward(self, anchor, positive, negative):
        anchor_embedding = self.feature_extractor(anchor)
        positive_embedding = self.feature_extractor(positive)
        negative_embedding = self.feature_extractor(negative)
        return anchor_embedding, positive_embedding, negative_embedding

# Use the TripletMarginLoss from the pytorch_metric_learning
criterion = losses.TripletMarginLoss(margin=1.0)

# Load a pre-trained model as the feature extractor
feature_extractor = fasterrcnn_resnet50_fpn(pretrained=True).backbone.body

# Create the triplet network and set up the training process
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TripletNetwork(feature_extractor).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
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

        for anchor_id in anchor_ids:
            positive_imgs = []
            negative_imgs = []
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

        positive_imgs = torch.stack(positive_imgs).to(device)
        negative_imgs = torch.stack(negative_imgs).to(device)

        optimizer.zero_grad()

        # TODO: Next task, fix the model output: AttributeError: 'collections.OrderedDict' object has no attribute 'shape'
        anchor_embeddings, positive_embeddings, negative_embeddings = model(anchor_imgs, positive_imgs, negative_imgs)
        loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    epoch_loss /= num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Validate the model ! Change to make it equal to the training part
    # model.eval()
    # with torch.no_grad():
    #     val_loss = 0.0
    #     num_batches = 0
    #     for anchor_imgs, anchor_cat_ids in val_dataloader:
    #         anchor_imgs = anchor_imgs.to(device)
    #         # anchor_cat_ids = [get_category_ids(coco_val, [img_id]) for img_id in anchor_img_ids]

    #         # Select a positive image
    #         shared_cat_id = np.random.choice(anchor_cat_ids)
    #         positive_img_id = np.random.choice(val_ann[str(shared_cat_id)])
    #         positive_img, _ = val_dataloader.dataset[val_dataloader.dataset.img_ids.index(positive_img_id)]

    #         # Select a negative image
    #         while True:
    #             negative_img_id = np.random.choice(val_dataloader.dataset.img_ids)
    #             # negative_cat_ids = get_category_ids(coco_val, [negative_img_id])
    #             negative_img, negative_cat_ids = val_dataloader.dataset[val_dataloader.dataset.img_ids.index(negative_img_id)]
    #             if not anchor_cat_ids.intersection(negative_cat_ids):
    #                 break

    #         positive_img = positive_img.to(device)
    #         negative_img = negative_img.to(device)

    #         anchor_embeddings, positive_embeddings, negative_embeddings = model(anchor_imgs, positive_img, negative_img)
    #         loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)

    #         val_loss += loss.item()
    #         num_batches += 1

    #     val_loss /= num_batches
    #     print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

# ! I'm not sure about this evaluation function
# # Evaluate query function
# def evaluate_query(query_img, query_img_id, database_imgs, database_img_ids, model, coco, top_k=5):
#     model.eval()
#     with torch.no_grad():
#         query_embedding = model.feature_extractor(query_img.unsqueeze(0).to(device)).squeeze()

#         query_cat_ids = get_category_ids(coco, [query_img_id])

#         database_embeddings = []
#         for img in database_imgs:
#             img_embedding = model.feature_extractor(img.unsqueeze(0).to(device)).squeeze()
#             database_embeddings.append(img_embedding)

#         database_embeddings = torch.stack(database_embeddings)
#         distances = torch.cdist(query_embedding.unsqueeze(0), database_embeddings)
#         sorted_indices = torch.argsort(distances)
#         top_k_indices = sorted_indices[0, :top_k].cpu().numpy()

#         correct = 0
#         for idx in top_k_indices:
#             database_img_id = database_img_ids[idx]
#             database_cat_ids = get_category_ids(coco, [database_img_id])
#             if query_cat_ids.intersection(database_cat_ids):
#                 correct += 1

#     return correct / top_k

# # Evaluate image retrieval function
# def evaluate_image_retrieval(query_loader, database_loader, model, coco):
#     correct = 0
#     total = 0

#     for query_img, query_img_id in query_loader:
#         total += 1
#         query_img = query_img.to(device)

#         database_imgs = [img for img, _ in database_loader.dataset]
#         database_img_ids = [img_id for _, img_id in database_loader.dataset]

#         correct += evaluate_query(query_img, query_img_id, database_imgs, database_img_ids, model, coco)

#     return correct / total

# # Evaluate the model using the evaluation functions
# query_dataloader = create_dataloaders(coco_val, val_img_dir, val_ann, batch_size=1)
# database_dataloader = create_dataloaders(coco_train, train_img_dir, database_ann, batch_size=1)

# retrieval_accuracy = evaluate_image_retrieval(query_dataloader, database_dataloader, model, coco_val)
# print("Image retrieval accuracy:", retrieval_accuracy)