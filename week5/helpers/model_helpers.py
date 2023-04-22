import fasttext
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import json
import datetime

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
class TripletNetworkITT(nn.Module):
    def __init__(self, txt_emb_model, embedding_dim):
        super(TripletNetworkITT, self).__init__()
        ## IMAGE MODEL ##
        # Load the pre-trained ResNet-50 model
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Show the model architecture
        print(resnet)
        # Remove the last fully connected layer while maintaining batch size

        self.img_embedder = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        print(self.img_embedder)

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
        pos_cap_embs = self.txt_embedder(pos_captions)
        neg_cap_emb = self.txt_embedder(neg_captions)
        # Project the text embeddings to the common space
        pos_cap_embs = self.txt_projection(pos_cap_embs)
        neg_cap_emb = self.txt_projection(neg_cap_emb)
        
        return anchor_img_emb, pos_cap_embs, neg_cap_emb

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
