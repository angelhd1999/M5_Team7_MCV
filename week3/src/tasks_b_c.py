# Global
import os
import glob
import json
import random
import string
import argparse
from PIL import Image
# import cv2
import numpy as np
import torch
import pycocotools.mask as cocomask
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.structures.boxes import BoxMode
from detectron2.utils.visualizer import Visualizer

# Task B
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from skimage import io
import datetime
import pickle

def setup_config_predef(model_name):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda"
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Set threshold for this model
    cfg.DATASETS.TEST = ("out_of_context",)
    cfg.MODEL.FP16_ENABLED = True # Enable mixed precision training for faster inference
    cfg.MODEL.DEVICE="cuda"
    return cfg

def run_inference_and_visualize(predictor, img_path):
    print("Running inference and visualizing prediction for image " + img_path)
    
    # Run inference
    img = Image.open(img_path)
    img = np.array(img)
    outputs = predictor(img)

    # Visualize prediction
    visualizer = Visualizer(img, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
    instances = outputs["instances"].to("cpu")
    vis_output = visualizer.draw_instance_predictions(instances)
    vis_img = vis_output.get_image()

    return vis_img

def get_mask_and_image(coco, class_name, save_path='./task_b/'):
    """
    Get a random image containing a specified class, save the image and its segmentation masks.

    Args:
        coco (COCO): COCO dataset instance.
        class_name (str): Name of the class for which segmentation masks should be saved.
        save_path (str, optional): Path to save the images and masks. Defaults to './test/'.
    """

    # Get the category ID for the specified class
    cat_ids = coco.getCatIds(catNms=[class_name])
    img_ids = coco.getImgIds(catIds=cat_ids)

    # Get a random image containing the specified class
    metadata = coco.loadImgs(img_ids[np.random.randint(0, len(img_ids))])[0]
    img = io.imread(metadata['coco_url'])

    # Save the image
    plt.imsave(f'{save_path}{class_name}.jpg', img)

    # Get the annotations for the image
    ann_ids = coco.getAnnIds(imgIds=metadata['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    # Initialize a binary mask of the same size as the image
    mask_combined = np.zeros((metadata['height'], metadata['width']))
    count = 0

    # Iterate through annotations and save individual masks
    for ann in anns:
        count += 1
        mask = coco.annToMask(ann)
        plt.imsave(f'{save_path}{class_name}_{count}.jpg', mask)
        mask_combined += mask

    # Save the combined binary mask
    plt.imsave(f'{save_path}{class_name}_binmask.jpg', mask_combined)

    # Return the image and the mask of the first instance
    return img, coco.annToMask(anns[0])

def crop_instance_from_image(image, mask, class_name, save_path):
    # Get the minimum and maximum row and column indices where the mask is 1
    rows, cols = np.where(mask == 1)
    top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()

    # Crop the image and the mask using the minimum and maximum indices
    cropped_image = image[top:bottom+1, left:right+1]
    cropped_mask = mask[top:bottom+1, left:right+1]

    # Create a new image with the same size as the cropped image and an additional alpha channel
    cropped_instance = np.zeros((*cropped_image.shape[:2], 4), dtype=np.uint8)

    # Copy the RGB channels from the cropped image to the cropped instance
    cropped_instance[..., :3] = cropped_image

    # Set the alpha channel to 255 (opaque) for pixels where the cropped mask is 1, and 0 (transparent) for the background
    cropped_instance[..., 3] = (cropped_mask == 1).astype(np.uint8) * 255

    # Save the cropped instance
    plt.imsave(f'{save_path}{class_name}_cropped.jpg', cropped_instance)

    return cropped_instance

def add_instance_to_image(image, instance, positions):
    edited_images = []

    # Create an alpha channel for the instance
    alpha = (instance[..., 3] != 0).astype(np.uint8) * 255

    for position in positions:
        y, x = position

        # Calculate the adjusted position and size for the instance
        y_min = max(0, y)
        x_min = max(0, x)
        y_max = min(image.shape[0], y + instance.shape[0])
        x_max = min(image.shape[1], x + instance.shape[1])

        # Adjust the instance position and size if it goes beyond the image boundaries
        instance_y_min = max(0, -y)
        instance_x_min = max(0, -x)
        instance_y_max = instance_y_min + y_max - y_min
        instance_x_max = instance_x_min + x_max - x_min

        # Place the instance on the image
        edited_image = image.copy()
        edited_image[y_min:y_max, x_min:x_max] = np.where(
            alpha[instance_y_min:instance_y_max, instance_x_min:instance_x_max][..., np.newaxis] == 255,
            instance[instance_y_min:instance_y_max, instance_x_min:instance_x_max, :3],
            edited_image[y_min:y_max, x_min:x_max]
        )

        edited_images.append(edited_image)

    return edited_images

def task_b(predictor, coco, class_name, least5_per_class_dict, random_positions_num, save_path):
    # Get the least coocurrent class
    least_coocurrent_class = least5_per_class_dict[class_name][0][0]

    # Get the image and mask of the class instance
    image, mask = get_mask_and_image(coco, class_name, save_path=save_path)

    # Crop the class instance from the image
    cropped_instance = crop_instance_from_image(image, mask, class_name, save_path=save_path)

    # Get a random image of the least coocurrent class
    least_coocurrent_image, _ = get_mask_and_image(coco, least_coocurrent_class, save_path=save_path)

    # Add the cropped instance to the least coocurrent image at random_positions_num random positions
    height, width, _ = least_coocurrent_image.shape
    instance_height, instance_width, _ = cropped_instance.shape
    positions = [
        (
            random.randint(0, max(0, height - instance_height)),
            random.randint(0, max(0, width - instance_width))
        )
        for _ in range(random_positions_num)
    ]

    edited_images = add_instance_to_image(least_coocurrent_image, cropped_instance, positions)

    img_original = f'{save_path}{class_name}.jpg'
    vis_img_original = f'{save_path}{class_name}_vis.jpg'
    plt.imsave(vis_img_original, run_inference_and_visualize(predictor, img_original))

    # Save the edited images
    for i, edited_image in enumerate(edited_images):
        i += 1
        img_name = f'{save_path}{class_name}_on_{least_coocurrent_class}_{i}.jpg'
        vis_img_name = f'{save_path}{class_name}_on_{least_coocurrent_class}_{i}_vis.jpg'
        plt.imsave(img_name, edited_image)
        plt.imsave(vis_img_name, run_inference_and_visualize(predictor, img_name))

def task_c(predictor, coco, class_name, save_path, random_positions_num):
    # Get the image and mask of the instance
    image, mask = get_mask_and_image(coco, class_name, save_path=save_path)

    # Crop the instance from the image
    cropped_instance = crop_instance_from_image(image, mask, class_name, save_path=save_path)

    # Generate random positions
    height, width, _ = image.shape
    instance_height, instance_width, _ = cropped_instance.shape
    positions = [
        (
            random.randint(0, max(0, height - instance_height)),
            random.randint(0, max(0, width - instance_width))
        )
        for _ in range(random_positions_num)
    ]

    img_original = f'{save_path}{class_name}.jpg'
    vis_img_original = f'{save_path}{class_name}_vis.jpg'
    plt.imsave(vis_img_original, run_inference_and_visualize(predictor, img_original))

    # Add the cropped instance to the original image at random positions
    edited_images = add_instance_to_image(image, cropped_instance, positions)

    # Save the edited images
    for i, edited_image in enumerate(edited_images):
        img_name = f'{save_path}{class_name}_with_cropped_instance_{i+1}.jpg'
        vis_img_name = f'{save_path}{class_name}_with_cropped_instance_{i+1}_vis.jpg'
        plt.imsave(img_name, edited_image)

        # Run inference and visualize the result
        vis_img = run_inference_and_visualize(predictor, img_name)
        plt.imsave(vis_img_name, vis_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Tasks B and C with the MaskRCNN or FasterRCNN model.")
    parser.add_argument('--model', default='MaskRCNN', choices=['MaskRCNN', 'FasterRCNN'], help="The object detection model to use.")
    parser.add_argument('--task_b', action='store_true', help="Perform Task B.")
    parser.add_argument('--task_c', action='store_true', help="Perform Task C.")
    parser.add_argument('--class_name', default='person', help="The class name to use for Task B and C.")
    parser.add_argument('--random_positions_num', type=int, default=3, help="The number of random positions for placing instances in Task B and C.")
    args = parser.parse_args()

    # Global parameters
    MODEL = args.model
    DO_TASK_B = args.task_b
    DO_TASK_C = args.task_c
    # Task B and C Parameters
    class_name = args.class_name
    random_positions_num = args.random_positions_num

    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # COCO setting
    path_to_coco = '../../../COCO'
    coco = COCO(f'{path_to_coco}/annotations/instances_val2017.json')
    # GENERAL: Load the model and the weights
    print(f"Loading models and weights from {MODEL} model") 
    if MODEL == 'FasterRCNN':
        cfg = setup_config_predef("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml") # Some options: "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" / "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    elif MODEL == 'MaskRCNN':
        cfg = setup_config_predef("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml") # Some options: "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" / "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    else:
        raise ValueError("MODEL should be 'MaskRCNN' or 'FasterRCNN'")
    predictor = DefaultPredictor(cfg)

    if DO_TASK_B:
        # Load the least5_per_class.pkl file
        with open('least5_per_class.pkl', 'rb') as f:
            least5_per_class_dict = pickle.load(f)

        # Create a folder to save the images using python, add a timestamp to the folder name
        experiment_path = f'./task_b/{timestamp}_{class_name}/'
        os.mkdir(experiment_path)
        task_b(predictor, coco, class_name, least5_per_class_dict, random_positions_num, experiment_path)

    if DO_TASK_C:
        # Create a folder to save the images using python, add a timestamp to the folder name
        experiment_path = f'./task_c/{timestamp}_{class_name}/'
        os.mkdir(experiment_path)    
        task_c(predictor, coco, class_name, experiment_path, random_positions_num)