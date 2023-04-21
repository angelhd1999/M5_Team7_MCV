import fasttext
import torch
import torch.nn as nn
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
class TripletNetwork(nn.Module):
    def __init__(self, feature_extractor):
        super(TripletNetwork, self).__init__()
        self.feature_extractor = feature_extractor

    def forward(self, anchor, positive, negative):
        anchor_embedding = self.feature_extractor(anchor)
        # if positive is None: # Added for evaluation
        #     return anchor_embedding
        positive_embedding = self.feature_extractor(positive)
        negative_embedding = self.feature_extractor(negative)
        return anchor_embedding, positive_embedding, negative_embedding

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
