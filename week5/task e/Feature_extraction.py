import torch
import wandb
import numpy as np
import umap
import matplotlib.pyplot as plt

from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from models import ImgEncoder, TextEncoder, LinearEncoder
from utils import decay_learning_rate, mpk
from dataset import COCOImagesAndCaptions

from pytorch_metric_learning import miners, losses, reducers

from sklearn.neighbors import KNeighborsClassifier

import os
import json
import numpy as np
from transformers import BertTokenizer, BertModel
from time import time

load_features = False
data_path = '/Users/ayanbanerjee/Downloads/coco/'
output_path = "./results/task_d"
os.makedirs(output_path, exist_ok=True)


def extract_features(data, tokenizer, model, device):
    TextFeatures = []
    with torch.no_grad():
        model.eval()
        start_global_time = time()
        for i, key in enumerate(data):
            sentences = []
            start = time()
            for sentence in key['sentences']:
                #print("{}, {}".format(i,sentence['raw']))
                x = sentence['raw']
                x = tokenizer(x, return_tensors="pt")
                x = x.to(device)
                x = model(**x)['last_hidden_state']
                #print(x)
                x = x.to("cpu")
                sentences.append(np.array(x.squeeze().numpy()))
            TextFeatures.append(sentences)
            end = time() - start
            print('Time spent in image id {}: {:.3f} seconds'.format(i,end))
            '''if i == 100:
                break'''
        end_global_time = time() - start_global_time
        print('The evaluation has taken {:.3f} seconds.'.format(end_global_time))

    #train_TextFeatures = np.array(train_TextFeatures)
    return TextFeatures

def BERT():
    with open(f'{data_path}/coco.json') as f:
        dataset = json.load(f)
    
    with open(f'{data_path}/train.json') as f:
        train_data = json.load(f)
    with open(f'{data_path}/val.json') as f:
        val_data = json.load(f)
    with open(f'{data_path}/test.json') as f:
        test_data = json.load(f)

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Initializing a model from the bert-base-uncased style configuration
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_model = bert_model.to(device)

    #ALL DATA
    print('Start extracting Text features from Dataset...')
    TextFeatures = extract_features(dataset['images'], bert_tokenizer, bert_model, device)

    #Save data
    with open('{}/bert_feats.npy'.format(output_path), "wb") as f:
        np.save(f, TextFeatures)       
    print('Text features saved.')

    #TRAIN DATA
    print('Start extracting Text features from Train dataset...')
    train_TextFeatures = extract_features(train_data, bert_tokenizer, bert_model, device)
    #Save data
    with open('{}/train_Textfeatures.npy'.format(output_path), "wb") as f:
        np.save(f, train_TextFeatures)       
    print('Text features from Train dataset saved.')
    # VALIDATION DATA  
    print('Start extracting Text features from Validation dataset...')
    val_TextFeatures = extract_features(val_data, bert_tokenizer, bert_model, device)
    #Save data
    with open('{}/val_Textfeatures.npy'.format(output_path), "wb") as f:
        np.save(f, val_TextFeatures)       
    print('Text features from Validation dataset saved.')
    print('Start extracting Text features from Test dataset...')
    # TEST DATA
    test_TextFeatures = extract_features(test_data, bert_tokenizer, bert_model, device)
    #Save data
    with open('{}/test_Textfeatures.npy'.format(output_path), "wb") as f:
        np.save(f, test_TextFeatures)       
    print('Text features from Test dataset saved.')
    return train_TextFeatures, val_TextFeatures, test_TextFeatures
    return TextFeatures


if __name__ == '__main__':
    if load_features:
        #train_TextFeatures = np.load('{}/train_Textfeatures.npy'.format(output_path), allow_pickle=True)
        #val_TextFeatures = np.load('{}/val_Textfeatures.npy'.format(output_path), allow_pickle=True)
        #test_TextFeatures = np.load('{}/test_Textfeatures.npy'.format(output_path), allow_pickle=True)
        TextFeatures = np.load('{}/bert_feats.npy'.format(output_path), allow_pickle=True)
    else:
        #train_TextFeatures, val_TextFeatures, test_TextFeatures = BERT()
        TextFeatures = BERT()
