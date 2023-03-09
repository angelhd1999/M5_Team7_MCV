import torch
import torch.nn as nn
from torchsummary import summary
from config import INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNEL
import torch.optim as optim


def default_loss():
    '''It returns a function that takes in a model and returns a loss function that takes in the model's
    output and the target
    
    Returns
    -------
        A function that takes in a model and returns a loss function that takes in the model's output and
    the target
    '''
    return nn.CrossEntropyLoss()


def default_optimizer(model, learning_rate = 0.001):
    '''It returns an Adam optimizer as default with the learning rate set to 0.001 and the model's
    parameters
    
    Parameters
    ----------
    model
        The model we're training
    learning_rate
        The learning rate for the optimizer
    
    Returns
    -------
        The optimizer is being returned.
    
    '''
    return optim.Adam(model.parameters(), lr = learning_rate)


def get_default_device():
    '''If a GPU is available, return the GPU device, else return the CPU device
    
    Returns
    -------
        The device object (GPU or CPU)
    '''
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def model_prep_and_summary(model, device):
    '''It moves the model to the GPU and prints the model summary.
    
    Parameters
    ----------
    model
        The model to be moved to GPU and summarized
    device
        the device to use for training (GPU or CPU) 
    '''
    # Define the model and move it to GPU:
    model = model
    model = model.to(device)
    print('Current device: ' + str(device))
    print('Is Model on CUDA: ' + str(next(model.parameters()).is_cuda))
    # Display model summary:
    summary(model, (INPUT_CHANNEL, INPUT_WIDTH, INPUT_HEIGHT))