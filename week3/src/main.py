import os
import glob
import json
import random
import string
from PIL import Image
import cv2
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

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="KITTI-MOTS Fine-tuning and Evaluation")

    parser.add_argument("--dataset_path", default="../../../mcv/datasets/out_of_context", type=str,
                        help="Path to the KITTI-MOTS dataset")
    parser.add_argument("--model", default="FasterRCNN", type=str, choices=['MaskRCNN', 'FasterRCNN'],
                        help="Model to use: 'MaskRCNN' or 'FasterRCNN'")
    parser.add_argument("--finetuning", action="store_true",
                        help="Enable fine-tuning of the model (provide --finetuning flag)")
    parser.add_argument("--finetuning_path", default="", type=str,
                        help="Path to the finetuned weights")
    parser.add_argument("--images_test_start", type=int, default=0, 
                        help="Start index for test images")
    parser.add_argument("--n_images_test", default=10, type=int,
                        help="Number of test images to use for inference and visualization")
    parser.add_argument("--n_workers", default=4, type=int,
                        help="Number of workers to use for data loading")

    return parser.parse_args()

args = parse_arguments()

DATASET_PATH = args.dataset_path
MODEL = args.model
FINETUNING = args.finetuning
FINETUNING_PATH = args.finetuning_path
N_IMAGES_TEST = args.n_images_test
IMAGES_TEST_START = args.images_test_start
N_WORKERS = args.n_workers

# Log the parameters (arguments)
print("Parameters:")
print(f"DATASET_PATH: {DATASET_PATH}")
print(f"MODEL: {MODEL}")
print(f"FINETUNING: {FINETUNING}")
print(f"FINETUNING_PATH: {FINETUNING_PATH}")
print(f"N_IMAGES_TEST: {N_IMAGES_TEST}")
print(f"IMAGES_TEST_START: {IMAGES_TEST_START}")
print(f"N_WORKERS: {N_WORKERS}")



print("Starting script")

# Enble CUDA
#print("Enabling CUDA")
#torch.cuda.set_device(0)

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


def run_inference(predictor, image_list):
    predictions = []

    for img_path in image_list:
        print("Running inference on image " + img_path)
        img = Image.open(img_path)
        img = np.array(img)
        outputs = predictor(img)
        predictions.append(outputs)

    return predictions

def visualize_predictions(predictions, image_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for i, (pred, img_path) in enumerate(zip(predictions, image_list)):
        print(f"Visualizing prediction {i + 1}/{len(image_list)}: {img_path}")
        img = cv2.imread(img_path)
        visualizer = Visualizer(img[:, :, ::-1],MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
        
        instances = pred["instances"].to("cpu")
        vis_output = visualizer.draw_instance_predictions(instances)
        
        vis_img = vis_output.get_image()[:, :, ::-1]
        output_file = os.path.join(output_dir, os.path.dirname(img_path).split('/')[-1] + '_' + os.path.basename(img_path))
        cv2.imwrite(output_file, vis_img)



print(f"Loading models and weights from {MODEL} model") 
if MODEL == 'FasterRCNN':
    cfg = setup_config_predef("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml") # Some options: "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" / "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
elif MODEL == 'MaskRCNN':
    cfg = setup_config_predef("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml") # Some options: "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" / "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
else:
    raise ValueError("MODEL should be 'MaskRCNN' or 'FasterRCNN'")

# Defining the output of the model
pool = string.ascii_letters + string.digits # Define the pool of characters to choose from
random_string = ''.join(random.choice(pool) for i in range(6)) # Generate a random string of length 6
if FINETUNING:
    finetuning_str = "finetuned"
else:
    finetuning_str = "pretrained"

predictor = DefaultPredictor(cfg)

print("Loading test images")
test_image_list = sorted(glob.glob(DATASET_PATH + '/*.jpg'))
print(test_image_list)

print(f"Running inference")
predictions = run_inference(predictor, test_image_list)
print("Inference done")

print("Visualizing predictions")
visualize_predictions(predictions, test_image_list,"output_visualizations")
print("Visualization done")
