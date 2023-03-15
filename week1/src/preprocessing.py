import csv
import os
import pandas as pd
from torchvision import transforms
import PIL


def build_annotation_dataframe(image_location, annot_location, output_csv_name):
    '''It takes a directory of folders of images, and creates a csv file with the file names, file paths,
    class names and class indices
    
    Parameters
    ----------
    image_location
        image directory path, e.g. r'.\data\\train'
    annot_location
        the directory where the csv file will be saved
    output_csv_name
        string of output csv file name, e.g. 'train.csv'
    
    Returns
    -------
        A dataframe with the file name, file path, class name and class index.
    
    '''
    class_lst = os.listdir(
        image_location)  # returns a LIST containing the names of the entries (folder names in this case) in the directory.
    class_lst.sort()  # IMPORTANT
    with open(os.path.join(annot_location, output_csv_name), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['file_name', 'file_path', 'class_name', 'class_index'])  # create column names
        for class_name in class_lst:
            # concatenates various path components with exactly one directory separator (‘/’) except the last path component.
            class_path = os.path.join(image_location, class_name)
            # get list of files in class folder
            file_list = os.listdir(class_path)
            for file_name in file_list:
                # concatenate class folder dir, class name and file name
                file_path = os.path.join(image_location, class_name, file_name)
                # write the file path and class name to the csv file
                writer.writerow(
                    [file_name, file_path, class_name, class_lst.index(class_name)])
    return pd.read_csv(os.path.join(annot_location, output_csv_name))


def check_annot_dataframe(annot_df):
    '''It takes a dataframe with two columns, one with class indices and one with class names, and returns
    a list of tuples with unique class indices and class names
    
    Parameters
    ----------
    annot_df
        the dataframe containing the annotations
    
    Returns
    -------
        A list of tuples.
    
    '''
    class_zip = zip(annot_df['class_index'], annot_df['class_name'])
    my_list = list()
    for index, name in class_zip:
        my_list.append(tuple((index, name)))
    unique_list = list(set(my_list))
    return unique_list


def transform_bilinear(output_img_width, output_img_height):
    '''"Resize and change the range of the input image to the specified height and width using bilinear
    interpolation."
    
    Parameters
    ----------
    output_img_width
        The width of the output image.
    output_img_height
        The height of the output image.
    
    Returns
    -------
        A callable object that, when called on an image, returns the resized image.
    
    '''
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # ? Are we sure images are in range [0, 1]? More info: https://discuss.pytorch.org/t/understanding-transform-normalize/21730
        transforms.Resize((output_img_width, output_img_height), interpolation=PIL.Image.NEAREST) # !I have change it to NEAREST instead of BILINEAR
    ])
    return image_transform