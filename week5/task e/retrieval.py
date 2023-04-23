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
from dataset import COCOImagesAndCaptionsBERT

from pytorch_metric_learning import miners, losses, reducers

from sklearn.neighbors import KNeighborsClassifier

import os
import pickle

dataset_path = '/Users/ayanbanerjee/Downloads/coco/'
output_path = "./results/task_d"
os.makedirs(output_path, exist_ok=True)
training = False

parser = ArgumentParser(
        description='Torch-based image classification system',
        formatter_class=ArgumentDefaultsHelpFormatter
)
parser.add_argument("anchor",
                    type=str,
                    help="Which modality to use as anchor")
parser.add_argument("num_epochs",
                    type=int,
                    help="Number of epochs")
parser.add_argument("lr",
                    type=float,
                    help="learning rate")
parser.add_argument("weight_decay",
                    type=float,
                    help="weight decay")
parser.add_argument("batch_size",
                    type=int,
                    help="batch size")
parser.add_argument("margin",
                    type=float,
                    help="change margin for triplet loss")
parser.add_argument("grad_clip",
                    type=float,
                    help="grad_clip")
parser.add_argument("gamma",
                    type=float,
                    help="Learning Rate Gamma")
parser.add_argument("mining_type",
                    type=str,
                    help="What mining modality to use (hard, semihard, easy, all)")

args = parser.parse_args()


def display_embeddings(text_embeddings, image_embeddings, text_labels, image_labels, epoch):
    reducer = umap.UMAP()
    n_text, _ = text_embeddings.shape
    n_imag, dim = image_embeddings.shape

    all_embeddings = np.vstack([text_embeddings, image_embeddings])
    all_embeddings = reducer.fit_transform(all_embeddings)

    text_embeddings = all_embeddings[:n_text]
    image_embeddings = all_embeddings[n_text:]

    plt.figure(dpi=300, figsize=(15, 15))
    plt.title(f"UMAP of the embedding space at epoch {epoch}")
    plt.scatter(text_embeddings[:, 0], text_embeddings[:, 1], color="orange", label="Text Embeddings")
    plt.scatter(image_embeddings[:, 0], image_embeddings[:, 1], color="cyan", label="Image Embeddings")

    for ii, label in enumerate(text_labels):
        plt.annotate(label, (text_embeddings[ii, 0], text_embeddings[ii, 1]), color="red", alpha=0.5)
    for ii, label in enumerate(image_labels):
        plt.annotate(label, (image_embeddings[ii, 0], image_embeddings[ii, 1]), color="blue", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path + f"/plots/plot_epoch_{epoch}.png")
    plt.close()


def validate(valid_dataloader, image_model, text_model, anchor, epoch, file_name):
    all_img_features = []
    all_txt_features = []

    image_model.eval()
    text_model.eval()

    with torch.no_grad():
        for i, (img_features, txt_features) in enumerate(valid_dataloader):
            img_features = img_features.to(device)  # (batch, ifeatures)
            txt_features = txt_features.to(device)  # (batch, ncaptions, tfeatures)

            batch_size, ncaptions, tfeatures = txt_features.shape

            # Reshape textual features so they are all encoded at once
            txt_features = txt_features.reshape((-1, tfeatures))

            img_encoded = image_model(img_features)
            txt_encoded = text_model(txt_features)

            all_img_features.append(img_encoded.detach().to("cpu").numpy())
            all_txt_features.append(txt_encoded.detach().to("cpu").numpy())

    all_img_features = np.vstack(all_img_features)
    all_txt_features = np.vstack(all_txt_features)

    all_img_labels = np.arange(len(all_img_features))
    all_txt_labels = np.arange(len(all_img_features)).repeat(
        len(all_txt_features) // len(all_img_features)
    )

    knn = KNeighborsClassifier(5, metric="euclidean")

    if anchor == "text":
        knn = knn.fit(all_img_features, all_img_labels)
        neighbors = knn.kneighbors(all_txt_features, return_distance=False)
        predictions = all_img_labels[neighbors]

        p1 = mpk(all_txt_labels, predictions, 1)
        p5 = mpk(all_txt_labels, predictions, 5)
    else:
        knn = knn.fit(all_txt_features, all_txt_labels)
        neighbors = knn.kneighbors(all_img_features, return_distance=False)
        predictions = all_txt_labels[neighbors]

        p1 = mpk(all_img_labels, predictions, 1)
        p5 = mpk(all_img_labels, predictions, 5)
    

    all_features = {'txt_features': all_txt_features, 'img_features': all_img_features, 'txt_labels': all_txt_labels, 'img_labels': all_img_labels}
    #Save data
    with open('{}/all_features_{}.pkl'.format(output_path, file_name), "wb") as f:
        pickle.dump(all_features, f)       
    print('Features and labels saved.')

    if not epoch % 100:
        display_embeddings(all_txt_features, all_img_features, all_txt_labels, all_img_labels, epoch)

    image_model.train()
    text_model.train()

    return p1, p5


if __name__ == '__main__':
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path + "/plots")
    wandb.init(
        dir=output_path,
        project="w5",
        entity="m5project",
        config=args.__dict__
    )
    reducer = reducers.MeanReducer()
    loss_func = losses.TripletMarginLoss(args.margin, reducer=reducer)
    # loss_func = nn.TripletMarginLoss(args.margin)
    miner = miners.TripletMarginMiner(args.margin, type_of_triplets=args.mining_type)

    train_set = FlickrImagesAndCaptionsBERT(dataset_path, "train")
    val_set = FlickrImagesAndCaptionsBERT(dataset_path, "val")
    test_set = FlickrImagesAndCaptionsBERT(dataset_path, "test")

    train_dataloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_dataloader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    test_dataloader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # TEXT & IMGS MODELS
    # image_model = ImgEncoder(embedding_size=64)
    # text_model = TextEncoder(embedding_size=64)

    image_model = LinearEncoder(4096, [384, 256])
    text_model = LinearEncoder(768, [384, 256])

    image_model.init_weights()
    text_model.init_weights()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_model.to(device)
    text_model.to(device)

    if training:
         # optimizer
        params = list(image_model.parameters())
        params += list(text_model.parameters())

        optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = ExponentialLR(optimizer, args.gamma)

        image_model.train()
        text_model.train()

        #p1, p5 = validate(val_dataloader, image_model, text_model, args.anchor, -1)

        # training loop
        iterations = 1
        for epoch in range(args.num_epochs):
            for i, (img_features, txt_features) in enumerate(train_dataloader):
                img_features = img_features.to(device)  # (batch, ifeatures)
                txt_features = txt_features.to(device)  # (batch, ncaptions, tfeatures)

                batch_size, ncaptions, tfeatures = txt_features.shape

                # Reshape textual features so they are all encoded at once
                txt_features = txt_features.reshape((-1, tfeatures))

                img_encoded = image_model(img_features)
                txt_encoded = text_model(txt_features)

                img_labels = torch.arange(batch_size)
                txt_labels = torch.arange(batch_size).repeat_interleave(ncaptions)

                # Create all training tuples according to modality anchor
                if args.anchor == "text":
                    tuples = miner(txt_encoded, txt_labels, img_encoded, img_labels)
                    loss = loss_func(
                        txt_encoded,
                        txt_labels,
                        tuples,
                        ref_emb=img_encoded,
                        ref_labels=img_labels
                    )
                else:
                    tuples = miner(img_encoded, img_labels, txt_encoded, txt_labels)
                    loss = loss_func(
                        img_encoded,
                        img_labels,
                        tuples,
                        ref_emb=txt_encoded,
                        ref_labels=txt_labels
                    )

                optimizer.zero_grad()

                loss.backward()
                # if args.grad_clip > 0:
                #     clip_grad_norm_(params, args.grad_clip)
                optimizer.step()
                iterations += 1

                wandb.log({
                    "step": iterations,
                    "train_loss": loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                })
            p1, p5 = validate(val_dataloader, image_model, text_model, args.anchor, epoch)
            wandb.log({
                "epoch": epoch,
                "p1": p1,
                "p5": p5,
            })
            scheduler.step()
        
        state_dict = [image_model.state_dict(), text_model.state_dict()]
        model_folder = str(output_path + "/models")
        os.makedirs(model_folder, exist_ok=True)
        torch.save(state_dict, f'{model_folder}/{args.anchor}_weights.pth')
    else:
        #LOAD PRETRAINED WEIGHTS
        state_dict =  torch.load('{}/models/image_weights.pth'.format(output_path))
        image_model.load_state_dict(state_dict[0])
        text_model.load_state_dict(state_dict[1])

        # optimizer
        params = list(image_model.parameters())
        params += list(text_model.parameters())

        optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = ExponentialLR(optimizer, args.gamma)

        image_model.train()
        text_model.train()

        p1, p5 = validate(train_dataloader, image_model, text_model, args.anchor, -1, 'train')
        p1, p5 = validate(val_dataloader, image_model, text_model, args.anchor, -1, 'validation')
        p1, p5 = validate(test_dataloader, image_model, text_model, args.anchor, -1, 'test')