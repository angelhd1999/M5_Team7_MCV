# Import the necessary packages
import pickle
from pathlib import Path
from tqdm.auto import notebook_tqdm
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from pytorch_metric_learning import losses
from pytorch_metric_learning import distances
from pytorch_metric_learning import losses
from pytorch_metric_learning import samplers

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import models
from torchvision.utils import make_grid

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, average_precision_score
from metrics import mpk, mAP

# Using a pretrained classifier for image retrieval
def resnet50():
    model = models.resnet50(pretrained=True, progress=False)
    model = nn.Sequential(*list(model.children())[:-1])
    return model

model = resnet50()

# Gathering the dataset
mit_split_data = Path('/Users/ayanbanerjee/Documents/m5/project/week1/MIT_split')
# Saving the extracted features
features = Path('./features')

# Basic transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Feature extraction with applied transformation
database = ImageFolder(str(mit_split_data / "train"), transform=transform)
queries = ImageFolder(str(mit_split_data / "test"), transform=transform)

# Gathering the labels
database_meta = [(x[0].split('/')[-1], x[1]) for x in database.imgs]
query_meta = [(x[0].split('/')[-1], x[1]) for x in queries.imgs]

# Storing the labels for further use
with (features / "database_metadata.pkl").open('wb') as d_meta:
    pickle.dump(database_meta, d_meta)
with (features / "query_metadata.pkl").open('wb') as q_meta:
    pickle.dump(query_meta, q_meta)

# Gathering database features
database_features = np.empty((len(database), 2048))
with torch.no_grad():
    for ii, (img, _) in enumerate(database):
        database_features[ii, :] = model(img.unsqueeze(0)).squeeze().numpy()
# Storing database features
with open(features / "database.npy", "wb") as d:
    np.save(d, database_features)

query_features = np.empty((len(queries), 2048))
with torch.no_grad():
    for ii, (img, _) in enumerate(queries):
        query_features[ii, :] = model(img.unsqueeze(0)).squeeze().numpy()

with open(features / "queries.npy", "wb") as q:
    np.save(q, query_features)


"""
Note: If you have already saved features then you can comment out the from line 42 to line 79. 
"""

# Uncomment the following lines for already saved features and labels
#with open(features / "queries.npy", "rb") as q:
#    query_features = np.load(q)
#with open(features / "database.npy", "rb") as d:
#    database_features = np.load(d)

#with (features / "database_metadata.pkl").open('rb') as fd, \
#        (features / "query_metadata.pkl").open('rb') as fq:
#    database_meta = pickle.load(fd)
#    query_meta = pickle.load(fq)

# Matching with KNN
database_labels = np.asarray([x[1] for x in database_meta])
query_labels = np.asarray([x[1] for x in query_meta])

knn = KNeighborsClassifier(n_neighbors=20, metric = "cosine")
knn = knn.fit(database_features, database_labels)
predictions = knn.predict(query_features)
pr_prob = knn.predict_proba(query_features)
neighbors = knn.kneighbors(query_features, return_distance=False)

# Calculating the performance
one_hot = np.zeros((predictions.shape[0], max(predictions) + 1), dtype=int)
one_hot[predictions] = 1

f1 = f1_score(query_labels, predictions, average="macro")
ap = average_precision_score(one_hot, pr_prob)

print("F1-Score:", f1)
print("Average Precision Score:", ap)

neighbors_labels = []
for i in range(len(neighbors)):
    neighbors_class = [database_meta[j][1] for j in neighbors[i]]
    neighbors_labels.append(neighbors_class)

query_labels = [x[1] for x in query_meta]

p_1 = mpk(query_labels,neighbors_labels, 1)
p_5 = mpk(query_labels,neighbors_labels, 5)

print('P@1=',p_1)
print('P@5=',p_5)

map = mAP(query_labels,neighbors_labels)
print('mAP=',map)











