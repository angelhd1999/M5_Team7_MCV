import fasttext
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import json
import datetime
import numpy as np

# Implement the triplet network model
class TripletNetworkITT(nn.Module):
    def __init__(self, txt_emb_model, embedding_dim, device):
        super(TripletNetworkITT, self).__init__()
        self.device = device
        ## IMAGE MODEL ##
        # Load the pre-trained ResNet-50 model
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove the last fully connected layer
        self.img_embedder = torch.nn.Sequential(*(list(resnet.children())[:-1]))

        ## TEXT MODEL ##
        # Set is_fasttext to True if the txt_emb_model is fasttext
        self.is_fasttext = True if txt_emb_model == 'fasttext' else False
        if self.is_fasttext:
            # Load the fasttext model
            print('Using fasttext')
            fasttext_model = fasttext.load_model('../../../mcv/m5/fasttext_wiki.en.bin')
            self.txt_embedder = fasttext_model.get_sentence_vector
        else:
            raise ValueError(f'BERT not implemented yet')
            print('Using BERT')
        
        ## COMMON SPACE PROJECTION ##
        # ? Image embedding projection, 2048 is the size of the output of the last layer of modified ResNet-50
        self.img_projection = nn.Linear(2048, embedding_dim)
        # ? Text embedding projection, 300 is the size of the output of the fasttext model
        self.txt_projection = nn.Linear(300, embedding_dim)

    def forward(self, anchor_imgs, pos_captions, neg_captions):
        # Print shapes
        # print(f'anchor_imgs.shape: {anchor_imgs.shape}') # ? anchor_imgs shape torch.Size([64, 3, 224, 224])
        ##* Get the image embeddings
        anchor_img_emb = self.img_embedder(anchor_imgs)
        # print(f'pre flatten anchor_img_emb.shape: {anchor_img_emb.shape}') # ? anchor_img_emb shape torch.Size([64, 2048, 1, 1])
        # Flatten the image embeddings
        anchor_img_emb = torch.flatten(anchor_img_emb, 1) 
        # print(f'post flatten anchor_img_emb.shape: {anchor_img_emb.shape}') # ? anchor_img_emb shape torch.Size([64, 2048])
        # Project the image embeddings to the common space
        anchor_img_emb = self.img_projection(anchor_img_emb)

        ##* Get the text embeddings
        # Get the text embeddings
        pos_cap_embs = [self.txt_embedder(caption) for caption in pos_captions]
        neg_cap_embs = [self.txt_embedder(caption) for caption in neg_captions]

        # Convert the lists of embeddings to tensors
        pos_cap_embs = torch.tensor(np.stack(pos_cap_embs), dtype=torch.float).to(self.device)
        neg_cap_embs = torch.tensor(np.stack(neg_cap_embs), dtype=torch.float).to(self.device)
        # print(f'pos_cap_embs.shape: {pos_cap_embs.shape}') # ? pos_cap_embs.shape: torch.Size([64, 300])
        # print(f'neg_cap_embs.shape: {neg_cap_embs.shape}') # ? neg_cap_embs.shape: torch.Size([64, 300])

        # Project the text embeddings to the common space
        pos_cap_embs = self.txt_projection(pos_cap_embs)
        neg_cap_embs = self.txt_projection(neg_cap_embs)

        return anchor_img_emb, pos_cap_embs, neg_cap_embs

class TripletNetworkTTI(nn.Module):
    def __init__(self, txt_emb_model, embedding_dim, device):
        super(TripletNetworkTTI, self).__init__()
        self.device = device
        ## IMAGE MODEL ##
        # Load the pre-trained ResNet-50 model
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove the last fully connected layer
        self.img_embedder = torch.nn.Sequential(*(list(resnet.children())[:-1]))

        ## TEXT MODEL ##
        # Set is_fasttext to True if the txt_emb_model is fasttext
        self.is_fasttext = True if txt_emb_model == 'fasttext' else False
        if self.is_fasttext:
            # Load the fasttext model
            print('Using fasttext')
            fasttext_model = fasttext.load_model('../../../mcv/m5/fasttext_wiki.en.bin')
            self.txt_embedder = fasttext_model.get_sentence_vector
        else:
            raise ValueError(f'BERT not implemented yet')
            print('Using BERT')
        
        ## COMMON SPACE PROJECTION ##
        # ? Image embedding projection, 2048 is the size of the output of the last layer of modified ResNet-50
        self.img_projection = nn.Linear(2048, embedding_dim)
        # ? Text embedding projection, 300 is the size of the output of the fasttext model
        self.txt_projection = nn.Linear(300, embedding_dim)

    def forward(self, anchor_captions, pos_imgs, neg_imgs):
        # Print shapes
        # print(f'anchor_imgs.shape: {anchor_imgs.shape}') # ? anchor_imgs shape torch.Size([64, 3, 224, 224])
        ##* Get the image embeddings
        pos_img_emb = self.img_embedder(pos_imgs)
        neg_img_emb = self.img_embedder(neg_imgs)
        # print(f'pre flatten anchor_img_emb.shape: {anchor_img_emb.shape}') # ? anchor_img_emb shape torch.Size([64, 2048, 1, 1])
        # Flatten the image embeddings
        pos_img_emb = torch.flatten(pos_img_emb, 1) 
        neg_img_emb = torch.flatten(neg_img_emb, 1) 
        # print(f'post flatten anchor_img_emb.shape: {anchor_img_emb.shape}') # ? anchor_img_emb shape torch.Size([64, 2048])
        # Project the image embeddings to the common space
        pos_img_emb = self.img_projection(pos_img_emb)
        neg_img_emb = self.img_projection(neg_img_emb)

        ##* Get the text embeddings
        # Get the text embeddings
        anchor_cap_embs = [self.txt_embedder(caption) for caption in anchor_captions]

        # Convert the lists of embeddings to tensors
        anchor_cap_embs = torch.tensor(np.stack(anchor_cap_embs), dtype=torch.float).to(self.device)

        # print(f'anchor_cap_embs.shape: {anchor_cap_embs.shape}')
        anchor_cap_embs = self.txt_projection(anchor_cap_embs)

        return anchor_cap_embs, pos_img_emb, neg_img_emb

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
