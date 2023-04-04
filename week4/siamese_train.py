#%matplotlib inline
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from siamese_network import SiameseNetwork,SiameseNetworkDataset,ContrastiveLoss

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Helper functions
def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

# Configuration Class
class Config():
    training_dir = "/Users/ayanbanerjee/Documents/m5/project/week1/MIT_split/train/"
    testing_dir = "/Users/ayanbanerjee/Documents/m5/project/week1/MIT_split/test/"
    train_batch_size = 64
    train_number_epochs = 5

# Using Image Folder Dataset
folder_dataset = dset.ImageFolder(root=Config.training_dir)

siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize(100),
                                                               transforms.RandomResizedCrop(100),
                                                               transforms.RandomHorizontalFlip(),
                                                               transforms.RandomRotation(10),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
                                       ,should_invert=False)

# Visualising some of the data
vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=1,
                        batch_size=8)
dataiter = iter(vis_dataloader)


example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0],example_batch[1]),0)
imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy())

# Training
train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=1,
                        batch_size=Config.train_batch_size)

# Model
net = SiameseNetwork() # For cpu
#net = SiameseNetwork().cuda() # For gpu
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.001)

counter = []
loss_history = [] 
iteration_number= 0

for epoch in range(0,Config.train_number_epochs):
    for i, data in enumerate(train_dataloader,0):
        img0, img1 , label = data
        img0, img1 , label = img0, img1 , label
        optimizer.zero_grad()
        output1,output2 = net(img0,img1)
        loss_contrastive = criterion(output1,output2,label)
        loss_contrastive.backward()
        optimizer.step()
        if i %10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
show_plot(counter,loss_history)

# Saving the model
torch.save(net.state_dict(), 'sm_constrastive.pt')
