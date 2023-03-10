import torch
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import random


class MITDataset(torch.utils.data.Dataset):
    def __init__(self, annot_df, transform=None):
        '''The __init__ function is used to initialize the class.
        
        Parameters
        ----------
        annot_df
            Dataframe containing the annotations.
        transform
            Transformation that will be applied to the image.
        '''
        self.annot_df = annot_df
        # root directory of images, leave "" if using the image path column in the __getitem__ method
        self.root_dir = "../../../mcv/datasets/MIT_split"
        self.transform = transform

    def __len__(self):
        '''This function returns the number of rows in the dataframe
        
        Returns
        -------
            The length of the dataframe
        
        '''
        # return length (numer of rows) of the dataframe
        return len(self.annot_df)

    def __getitem__(self, idx):
        '''The function takes in an index and returns the image, class name and class index
        
        Parameters
        ----------
        idx
            index of the image in the dataset
        
        Returns
        -------
            image, class_name, class_index
        
        '''
        # use image path column (index = 1) in csv file
        image_path = self.annot_df.iloc[idx, 1]
        image = cv2.imread(image_path)  # read image by cv2
        # convert from BGR to RGB for matplotlib
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # use class name column (index = 2) in csv file
        class_name = self.annot_df.iloc[idx, 2]
        # use class index column (index = 3) in csv file
        class_index = self.annot_df.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        # when accessing an instance via index, 3 outputs are returned - the image, class name and class index
        return image, class_name, class_index

    def visualize(self, number_of_img=10, output_width=12, output_height=6):
        '''It takes a random set of images from the dataset, gets the image, class name and class index, and
        plots it on a subplot
        
        Parameters
        ----------
        number_of_img, optional
            number of images to be displayed, defaults to 10 (optional)
        output_width, optional
            The width of the output image, defaults to 12 (optional)
        output_height, optional
            The height of the output image, defaults to 6 (optional)
        
        '''
        plt.figure(figsize=(output_width, output_height))
        for i in range(number_of_img):
            idx = random.randint(0, len(self.annot_df))
            image, class_name, class_index = self.__getitem__(idx)
            ax = plt.subplot(2, 5, i+1)  # create an axis
            # create a name of the axis based on the img name
            ax.title.set_text(class_name + '-' + str(class_index))
            if self.transform == None:
                plt.imshow(image)
            else:
                plt.imshow(image.permute(1, 2, 0))


def create_validation_dataset(dataset, validation_proportion):
    '''It takes a dataset and a proportion of the dataset to be used as a validation set, and returns a
    tuple of two datasets, the first being the training set and the second being the validation set
    
    Parameters
    ----------
    dataset
        The dataset to be split
    validation_proportion
        The proportion of the dataset that you want to be the validation set
    
    Returns
    -------
        The dataset and validation set are being returned.
    
    '''
    if (validation_proportion > 1) or (validation_proportion < 0):
        return "The proportion of the validation set must be between 0 and 1"
    else:
        dataset_size = int((1 - validation_proportion) * len(dataset))
        validation_size = len(dataset) - dataset_size
        print(dataset_size, validation_size)
        dataset, validation_set = torch.utils.data.random_split(dataset, [dataset_size, validation_size])
        return dataset, validation_set