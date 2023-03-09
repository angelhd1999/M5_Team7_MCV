import pathlib
import os
import sys
from torch.utils.data import DataLoader
import torch
import random

from config import TRAIN_DATA_LOC, TEST_DATA_LOC, ANNOT_LOC, MODEL_SAVE_LOC, REPORT_SAVE_LOC
from config import INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNEL, BATCH_SIZE, NUM_WORKERS
from preprocessing import build_annotation_dataframe, check_annot_dataframe, transform_bilinear
from model.dataset import MITDataset, create_validation_dataset
import model.cnn_model as cnn_model
import model.modelling_config as modelling_config
import model.custom_loss_function as custom_loss_function
from postprocessing import save_model_with_timestamp, save_csv_with_timestamp, calculate_model_performance, generate_fn_cost_matrix, generate_fp_cost_matrix


import importlib
import model.dataset
import model.cnn_model as cnn_model
import model.custom_loss_function as custom_loss_function
import config
import postprocessing
importlib.reload(model.dataset)
importlib.reload(model.cnn_model)
importlib.reload(model.modelling_config)
importlib.reload(model.custom_loss_function)
importlib.reload(config)
importlib.reload(postprocessing)
from model.dataset import MITDataset, create_validation_dataset
import model.cnn_model as cnn_model
import model.modelling_config as modelling_config
import config
from postprocessing import save_model_with_timestamp, save_csv_with_timestamp


print(torch.backends.cudnn.enabled)
print(torch.cuda.is_available())

import wandb

wandb.login()

config = {
    "epochs" : 10,
    "batch_size" : 16,
    "learning_rate": 0.001
}

run = wandb.init(project="mit_cnn", reinit="True", config=config)

# Creating and Preprocessing input data
train_df = build_annotation_dataframe(image_location=TRAIN_DATA_LOC, annot_location=ANNOT_LOC, output_csv_name='train.csv')
test_df = build_annotation_dataframe(image_location=TEST_DATA_LOC, annot_location=ANNOT_LOC, output_csv_name='test.csv')
class_names = list(train_df['class_name'].unique())
print(class_names)
print(check_annot_dataframe(train_df))
print(check_annot_dataframe(test_df))

# Creating the train, test and validation datasets
image_transform = transform_bilinear(INPUT_WIDTH, INPUT_HEIGHT)
main_dataset = MITDataset(annot_df = train_df, transform=image_transform)
train_dataset, validation_dataset = create_validation_dataset(main_dataset, validation_proportion=0.2)
print('Train set size: ', len(train_dataset))
print('Validation set size: ', len(validation_dataset))
test_dataset = MITDataset(annot_df = test_df, transform=image_transform)
print('Test set size: ', len(test_dataset))

# Configuring the DataLoaders
train_loader = DataLoader(train_dataset, batch_size = config["batch_size"], shuffle=True, num_workers = NUM_WORKERS)
val_loader = DataLoader(validation_dataset, batch_size = config["batch_size"], shuffle=True, num_workers = NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size = config["batch_size"], shuffle=True, num_workers = NUM_WORKERS)

# Training and Exporting the CNN model
model = cnn_model.MyCnnModel()
device = modelling_config.get_default_device()
modelling_config.model_prep_and_summary(model, device)
criterion = modelling_config.default_loss()
optimizer = modelling_config.default_optimizer(model = model, learning_rate = config["learning_rate"])
num_epochs = config["epochs"]

# get training results
trained_model, train_result_dict = cnn_model.train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs)
cnn_model.visualize_training(train_result_dict)

#saving the model and report for further use
save_model_with_timestamp(trained_model, MODEL_SAVE_LOC)
save_csv_with_timestamp(train_result_dict, REPORT_SAVE_LOC)

#Testing the model
trained_model_list = os.listdir(MODEL_SAVE_LOC)
MODEL_10_EPOCH_PATH = os.path.join(MODEL_SAVE_LOC, trained_model_list[0])
MODEL_10_EPOCH = cnn_model.MyCnnModel()
device = cnn_model.get_default_device()
print(MODEL_10_EPOCH_PATH)
MODEL_10_EPOCH.load_state_dict(torch.load(MODEL_10_EPOCH_PATH))

# check accuracy on test set
y_pred, y_true = cnn_model.infer(model = MODEL_10_EPOCH, device = device, data_loader = test_loader)
confusion_matrix, class_metrics, overall_metrics = calculate_model_performance(y_pred, y_true, class_names = class_names)

print(confusion_matrix)
print(class_metrics)
print(overall_metrics)





