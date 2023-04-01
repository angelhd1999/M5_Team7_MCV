# Import the necessary packages
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pickle
from metrics import mpk, mAP
import time

"""
FAISS is a feature matching strategy similar to KNN except the indexing.
"""

# Building index for FAISS
def build_index(model, train_dataset, d=32):
    index = faiss.IndexFlatL2(d)  # build the index

    xb = np.empty((len(train_dataset), d))
    find_in_train = dict()
    with torch.no_grad():
        model.eval()
        for ii, (data, label) in enumerate(train_dataset):
            find_in_train[ii] = (data, label)
            xb[ii, :] = model(data.unsqueeze(0)).squeeze().detach().numpy()

    xb = np.float32(xb)
    index.add(xb)  # add vectors to the index

    return index, find_in_train

# Creating the model
class EmbeddingLayer(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.linear = nn.Linear(2048, embed_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.squeeze(-1).squeeze(-1)
        x = self.activation(x)
        x = self.linear(x)
        return x
    
def resnet50(embed_size):
    embed = EmbeddingLayer(embed_size)
    model = models.resnet50(pretrained=True, progress=False)
    model = nn.Sequential(*list(model.children())[:-1], embed)
    return model

classes = ['Open Country', 'Coast', 'Forest', 'Highway', 'Inside City', 'Mountain', 'Street', 'Tall Building']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Basic transformations
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Gathering the dataset
mit_split_data = Path('/Users/ayanbanerjee/Documents/m5/project/week1/MIT_split')
EMBED_SHAPE = 32

# Feature extraction with applied transformation
database = ImageFolder(str(mit_split_data / "train"), transform=transforms)
queries = ImageFolder(str(mit_split_data / "test"), transform=transforms)

model = resnet50(EMBED_SHAPE)
model = model[:9]
index, find_in_train = build_index(model, database, d=2048)

k = 5 # As we are comparing with KNN, we are keeping all the setup same.
query_data = np.empty((len(queries), 2048))

pred_index = list()
gt_index = list()
metrics = list() # Helps to calculate all the metrics at once
time_list = [] # To compare which one is faster

# Feature Extraction
with torch.no_grad():
    for ii, (img, label) in enumerate(queries):
        xq = model(img.unsqueeze(0)).squeeze().numpy()
        xq = np.float32(xq)
        start = time.time()
        met, pred_in = index.search(np.array([xq]), k)
        end = time.time()
        pred_index.append(pred_in)
        gt_index.append(label)
        metrics.append(met)

# Some Visualizations
PLOT = True
if PLOT:
    plot_samples = 3
    fig, axs = plt.subplots(plot_samples, k)

    print(f"first {plot_samples}-th samples: ", pred_index[:plot_samples])
    for row in range(plot_samples):
        axs[row, 0].imshow(queries[row][0].permute((1, 2, 0)).numpy())  # plots query img
        for column in range(1, k):
            img_aux = find_in_train[pred_index[row][0][column]][0].permute((1, 2, 0))
            axs[row, column].imshow(img_aux.numpy())
            print(f"for img {row}, nn id: {pred_index[row][0][column]}")

    plt.title(f'{k} nearest imgs for firts {plot_samples}-th images (FAISS)')
    plt.savefig("./results/faiss.png")

SLIDES = True
if SLIDES:
    for xz in range(len(pred_index)):
        labels_list_auxz = pred_index[xz][0]
        for xy in range(len(labels_list_auxz)):
            auxxy = labels_list_auxz[xy]
            print(f"query_{xz}_k{xy}: {classes[find_in_train[auxxy][1]]}")
            plt.imsave(f"./results/query_{xz}_k{xy}.png",
                           find_in_train[auxxy][0].permute((1, 2, 0)).numpy())
            
# Quantitative Analysis
quant = list()

for jj, (pd_labels, gt_labs) in enumerate(zip(pred_index, gt_index)):
    id_nn = pd_labels[0]  # 1st nn
    aux = list()
    for ll in id_nn:
        aux.append(find_in_train[ll][1])
    quant.append(aux)

p_1 = mpk(gt_index, quant, 1)
p_5 = mpk(gt_index, quant, 5)
print('P@1={:.3f}'.format(p_1*100))
print('P@5={:.3f}'.format(p_5*100))

map = mAP(gt_index, quant)
print('mAP={:.3f}'.format(map*100))