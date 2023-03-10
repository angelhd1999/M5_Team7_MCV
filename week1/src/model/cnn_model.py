import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import wandb

class MyCnnModel(nn.Module):
    def __init__(self):
        super(MyCnnModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=1, padding=0)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, padding=0)

        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Define a max pooling layer to use repeatedly in the forward function
        # The role of pooling layer is to reduce the spatial dimension (H, W) of the input volume for next layers.
        # It only affects weight and height but not depth.
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.globalpool = nn.AdaptiveAvgPool2d((1,1))

        # output shape of maxpool3 is 64*28*28
        self.fc14 = nn.Linear(128, 256)
        # output of the final DC layer = 8 = number of classes
        self.fc16 = nn.Linear(256, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x1 = self.maxpool(x)

        x = F.relu(self.conv2(x))
        x2 = self.maxpool(x)
        concat1 = torch.add(x1, x2)

        x = F.relu(self.conv3(concat1))
        x3 = self.maxpool(x)
        x = F.relu(self.conv4(x))
        x4 = self.maxpool(x)
        concat2 = torch.add(x3, x4)

        x = F.relu(self.conv5(concat2))
        x5 = self.maxpool(x)
        x = self.globalpool(x5)

        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc14(x))
        x = F.dropout(x, 0.5)
        x = self.fc16(x)
        return x

def train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    '''We train the model for a number of epochs, and for each epoch we iterate through the training and
    validation datasets, calculate the loss and accuracy, and log the results to W&B
    
    Parameters
    ----------
    model
        the model to train
    device
        the device to run the training on (CPU or GPU)
    train_loader
        the training data loader
    val_loader
        validation data loader
    criterion
        the loss function
    optimizer
        The optimizer used to train the model
    num_epochs, optional
        number of epochs to train for, defaults to 5 (optional)
    
    Returns
    -------
        The model and the train_result_dict
    
    '''
    # device = get_default_device()
    model = model.to(device)
    train_result_dict = {'epoch': [], 'train_loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'time': []}
    wandb.watch(model, criterion, log="all", log_freq=10) # WANDB WATCH
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = 0.0
        correct = 0
        total = 0
        val_correct = 0
        val_total = 0
        model.train()  # set the model to training mode, parameters are updated
        for i, data in enumerate(train_loader, 0):
            image, class_name, class_index = data
            image = image.to(device)
            class_index = class_index.to(device)
            optimizer.zero_grad()  # zero the parameter gradients
            outputs = model(image)  # forward propagation
            loss = criterion(outputs, class_index)  # loss calculation
            loss.backward()  # backward propagation
            optimizer.step()  # params update
            train_loss += loss.item()  # loss for each minibatch
            _, predicted = torch.max(outputs.data, 1)
            total += class_index.size(0)
            correct += (predicted == class_index).sum().item()
        train_accuracy = round(float(correct)/float(total)*100, 2) # ! We have put train_accuracy out of the loop

        # Here evaluation is combined together with
        val_loss = 0.0
        model.eval()  # set the model to evaluation mode, parameters are frozen
        for i, data in enumerate(val_loader, 0):
            image, class_name, class_index = data
            image = image.to(device)
            class_index = class_index.to(device)
            outputs = model(image)
            loss = criterion(outputs, class_index)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += class_index.size(0)
            val_correct += (predicted == class_index).sum().item()
        val_accuracy = round(float(val_correct)/float(val_total)*100, 2)

        # print statistics every 1 epoch
        # divide by the length of the minibatch because loss.item() returns the loss of the whole minibatch
        train_loss_result = round(train_loss / len(train_loader), 3)
        val_loss_result = round(val_loss / len(val_loader), 3)

        epoch_time = round(time.time() - start_time, 1)
        # add statistics to the dictionary:
        train_result_dict['epoch'].append(epoch + 1)
        train_result_dict['train_loss'].append(train_loss_result)
        train_result_dict['val_loss'].append(val_loss_result)
        train_result_dict['accuracy'].append(train_accuracy)
        train_result_dict['val_accuracy'].append(val_accuracy)
        train_result_dict['time'].append(epoch_time)
        wandb.log({"train_loss":train_loss_result, "acc":train_accuracy, "val_loss": val_loss_result, "val_acc": val_accuracy}, step=epoch)
        print(f'''Epoch {epoch+1} 
            \t Training Loss: {train_loss_result} 
            \t Validation Loss: {val_loss_result} 
            \t Epoch Train Accuracy (%): {train_accuracy}
            \t Epoch Validation Accuracy (%): {val_accuracy} 
            \t Epoch Time (s): {epoch_time}
        ''')
    # return the trained model and the loss dictionary
    return model, train_result_dict


def visualize_training(train_result_dictionary):
    '''It takes a dictionary of training results and plots the training and validation loss and accuracy in
    a single plot
    
    Parameters
    ----------
    train_result_dictionary
        This is the dictionary that contains the training results
    
    '''
    # Define Data
    df = pd.DataFrame(train_result_dictionary)
    x = df['epoch']
    data_1 = df['train_loss']
    data_2 = df['val_loss']
    data_3 = df['accuracy']
    data_4 = df['val_accuracy']

    # Create Plot
    fig, ax1 = plt.subplots(figsize=(7, 7))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(x, data_1, color='red', label='training loss')
    ax1.plot(x, data_2, color='blue', label='validation loss')

    # Adding Twin Axes
    ax2 = ax1.twinx()
    ax2.plot(x, data_3, color='green', label='Training Accuracy')
    ax2.plot(x, data_4, color='orange', label='Validation Accuracy')

    # Add label
    plt.ylabel('Accuracy')
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc='upper center')

    # Show plot
    plt.show()


def infer(model, device, data_loader):
    '''> The function takes a trained model, a device, and a data loader as input, and returns the
    predicted class indices and the true class indices of the data loader
    
    Parameters
    ----------
    model
        the model to be evaluated
    device
        the device to run the model on (CPU or GPU)
    data_loader
        the data loader for the test set
    
    Returns
    -------
        The predicted class indices and the true class indices
    '''
    model = model.to(device)
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data in data_loader:
            image, class_name, class_index = data
            image = image.to(device)
            class_index = class_index.to(device)
            outputs = model(image)
            outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(outputs)
            class_index = class_index.data.cpu().numpy()
            y_true.extend(class_index)
    return y_pred, y_true


def infer_single_image(model, device, image_path, transform):
    '''It takes an image, transforms it, and then passes it through the model to get a predicted class
    index.
    
    Parameters
    ----------
    model
        the trained model
    device
        the device to run the model on (CPU or GPU)
    image_path
        the path to the image you want to classify
    transform
        This is the transformation that we applied to the images in the training set.
    
    Returns
    -------
        The predicted class index of the image.
    
    '''
    # Prepare the Image
    image = cv2.imread(image_path)  # read image by cv2
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_transformed = transform(image)
    plt.imshow(image_transformed.permute(1, 2, 0))
    image_transformed_sq = torch.unsqueeze(image_transformed, dim=0)

    # Inference
    model.eval()
    with torch.no_grad():
        image_transformed_sq = image_transformed_sq.to(device)
        output = model(image_transformed_sq)
        _, predicted_class_index = torch.max(output.data, 1)
    print(f'Predicted Class Index: {predicted_class_index}')
    return predicted_class_index
