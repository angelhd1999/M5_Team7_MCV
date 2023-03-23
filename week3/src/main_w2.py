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

    parser.add_argument("--dataset_path", default="../../../KITTI-MOTS", type=str,
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

# Previous mapping from KITTI-MOTS to COCO classes -> nan on Cars
# class_mapping_k_to_c = {  # Mapping KITTI-MOTS class indices to COCO class indices
#     1: 2, # KITTI-MOTS class 1 is "car" when read, so we map it to COCO class 2
#     2: 1 # KITTI-MOTS class 2 is "person" when read, so we map it to COCO class 1
# }

if FINETUNING:
    # New mapping from KITTI-MOTS to COCO classes -> nan on Pedestrians
    class_mapping_k_to_c = {  # Mapping KITTI-MOTS class indices to COCO class indices
        1: 2, # KITTI-MOTS class 1 is "car" when read, so we map it to COCO class 2
        2: 0 # KITTI-MOTS class 2 is "person" when read, so we map it to COCO class 1
    }
else:
    class_mapping_k_to_c = {  # Mapping KITTI-MOTS class indices to COCO class indices
        1: 2, # KITTI-MOTS class 1 is "car" when read, so we map it to COCO class 2
        2: 0 # KITTI-MOTS class 2 is "person" when read, so we map it to COCO class 1
    }

class_mapping_c_to_k = {  # Mapping COCO class indices to KITTI-MOTS class indices
    2: 2, # COCO class 2 is "car" so we map it to KITTI-MOTS class 0 (defined at thing_classes)
    0: 0 # COCO class 1 is "person" so we map it to KITTI-MOTS class 1 (defined at thing_classes)
} 
ignore_class = 1

# Add all other COCO classes to the mapping dictionary as an "Ignore" class
for i in range(80):
    if i not in class_mapping_c_to_k:
        class_mapping_c_to_k[i] = ignore_class

print("Starting script")

# Enble CUDA
#print("Enabling CUDA")
#torch.cuda.set_device(0)

def get_kitti_mots_dicts(data_path, mode, model):
    # assert mode in ['train_gen', 'val_gen'], "mode should be 'train_gen' or 'val_gen'"
    
    image_path = os.path.join(data_path, mode, 'image_02')
    # instance_path = os.path.join(data_path, 'instances') # ! Not used
    instance_txt_path = os.path.join(data_path, 'instances_txt' + '_' + mode)

    dataset_dicts = []

    for seq in sorted(os.listdir(image_path)):
        images = sorted(glob.glob(os.path.join(image_path, seq, '*.png')))
        # instances = sorted(glob.glob(os.path.join(instance_path, seq, '*.png'))) # ! Not used
        instance_txt = os.path.join(instance_txt_path, seq + '.txt')
        
        # Log progress
        print("Loading " + str(len(images)) + " images from sequence " + seq)

        with open(instance_txt, 'r') as f:
            instance_data = f.readlines()

        # for idx, (img_path, inst_path) in enumerate(zip(images, instances)): # ! Old line
        for idx, img_path in enumerate(images):
            record = {}

            record["file_name"] = img_path
            record["image_id"] = idx

            # Initializing height and width with None
            # in_height = 375 # None
            # in_width = 1242 # None
            # record["height"] = in_height
            # record["width"] = in_width
            # default_size = True
            
            # Parse instance data
            objs = []
            for line in instance_data:
                values = line.strip().split(' ')
                frame_id, class_instance_ids, class_id_re, h, w, rle_encoding = values
                
                if int(frame_id) != idx: # ? To not check unnecessary lines
                    continue

                h = int(h)
                w = int(w)
                record["height"] = h
                record["width"] = w

                # Set height and width only once 
                # if in_height is None and in_width is None: # ! This is never occuring now
                # if default_size:
                #     in_height = h
                #     in_width = w
                #     record["height"] = in_height
                #     record["width"] = in_width
                #     default_size = False
                
                # Parse class and instance id
                class_id = int(class_instance_ids) // 1000
                class_id_re = int(class_id_re) # ? It's redundant, but we'll keep it for now
                instance_id = int(class_instance_ids) % 1000
                
                # Decode RLE mask encoding
                rle = {'size': [h, w], 'counts': rle_encoding}
                binary_mask = cocomask.decode(rle)

                if class_id not in class_mapping_k_to_c:
                    continue
                # Map class id to COCO class id
                class_id = class_mapping_k_to_c[class_id]
                if np.sum(binary_mask) == 0:
                    continue
                    
                # Compute the bounding box from the mask
                y, x = np.where(binary_mask == 1)
                bbox = [int(np.min(x)), int(np.min(y)), int(np.max(x) - np.min(x)), int(np.max(y) - np.min(y))]

                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours = [list(np.squeeze(contour).flatten().astype(float)) for contour in contours if len(contour) > 3]

                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": class_id,
                    # "instance_id": instance_id,
                }
                if (model == 'MaskRCNN'):
                    shape = [(int(x), int(y)) for x, y in zip(x, y)]
                    shape = [s for x in shape for s in x]
                    #obj["segmentation"] = [shape]
                    obj["segmentation"] = contours

                objs.append(obj)

            # If there are no objects in the image, we skip it
            if len(objs) == 0:
                continue
            record["annotations"] = objs
            # record["sem_seg_file_name"] = inst_path # ? Not sure if we need this
            dataset_dicts.append(record)

    return dataset_dicts


def setup_config_predef(model_name):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda"
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Set threshold for this model
    cfg.DATASETS.TEST = ("kitti_mots_testing",)
    cfg.MODEL.FP16_ENABLED = True # Enable mixed precision training for faster inference
    cfg.MODEL.DEVICE="cuda"
    return cfg

def setup_config_finetuning(model_name, train_dataset_name, val_dataset_name):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda"
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.VAL = (val_dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.FP16_ENABLED = True  # Enable mixed precision training for faster inference
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # You can adjust this value depending on your GPU memory
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Number of classes in KITTI-MOTS dataset (excluding the ignore class)
    cfg.MODEL.DEVICE="cuda"


    # Set up the training parameters
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 1000 # Prev: 1000 # You can adjust the number of iterations based on your needs
    cfg.SOLVER.STEPS = []  # Do not decay learning rate
    cfg.SOLVER.GAMMA = 0.05
    cfg.SOLVER.CHECKPOINT_PERIOD = 500  # Save a checkpoint every 500 iterations
    cfg.TEST.EVAL_PERIOD = 100  # Evaluate the model every 500 iterations

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

def visualize_predictions(predictions, image_list, metadata, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for i, (pred, img_path) in enumerate(zip(predictions, image_list)):
        print(f"Visualizing prediction {i + 1}/{len(image_list)}: {img_path}")
        img = cv2.imread(img_path)
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        
        instances = pred["instances"].to("cpu")
        instances.pred_classes = torch.tensor([class_mapping_c_to_k[c.item()] for c in instances.pred_classes])  # Apply the class mapping
        filtered_instances = instances[instances.pred_classes != ignore_class]  # Filter out the "ignore" class
        vis_output = visualizer.draw_instance_predictions(filtered_instances)
        
        vis_img = vis_output.get_image()[:, :, ::-1]
        output_file = os.path.join(output_dir, os.path.dirname(img_path).split('/')[-1] + '_' + os.path.basename(img_path))
        cv2.imwrite(output_file, vis_img)

print("Loading KITTI-MOTS dataset")
# Register KITTI-MOTS dataset
for d in ["train_gen", "val_gen"]:
    print("Registering KITTI-MOTS " + d + " dataset")
    DatasetCatalog.register("kitti_mots_" + d, lambda d=d: get_kitti_mots_dicts(DATASET_PATH, d, MODEL) )
    MetadataCatalog.get("kitti_mots_" + d).set(thing_classes=["Pedestrian", "Ignore", "Car"])

# Get the metadata for the KITTI-MOTS training dataset, used for visualizing predictions
kitti_mots_metadata = MetadataCatalog.get("kitti_mots_val_gen")

print(f"Loading models and weights from {MODEL} model") 
if MODEL == 'FasterRCNN':
    if FINETUNING:
        cfg = setup_config_finetuning("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml", "kitti_mots_train_gen", "kitti_mots_val_gen")
    else:
        cfg = setup_config_predef("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml") # Some options: "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" / "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
elif MODEL == 'MaskRCNN':
    if FINETUNING:
        cfg = setup_config_finetuning("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml", "kitti_mots_train_gen", "kitti_mots_val_gen")
    else:
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

if FINETUNING:
    if FINETUNING_PATH != "":
        cfg.MODEL.WEIGHTS = os.path.join(FINETUNING_PATH, f"model_final.pth")
    else:
        # Create a directory to store the output of the model
        cfg_output_dir = f"./output_{MODEL}_{finetuning_str}_{random_string}"
        os.makedirs(cfg_output_dir, exist_ok=True)
        cfg.OUTPUT_DIR = cfg_output_dir

        # Create a DefaultTrainer
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)

        # Train the model
        trainer.train()
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, f"model_final.pth")

predictor = DefaultPredictor(cfg)

print("Loading test images")
test_image_list = sorted(glob.glob(DATASET_PATH + '/testing/image_02/*/*.png'))

# Calculate start and end indices for the images you want to process
images_test_end = min(IMAGES_TEST_START + N_IMAGES_TEST, len(test_image_list))

shortened_test_image_list = test_image_list[IMAGES_TEST_START:images_test_end]

print(f"Running inference on images {IMAGES_TEST_START} to {images_test_end-1}")

print("Running inference on " + str(len(shortened_test_image_list)) + " images")
predictions = run_inference(predictor, shortened_test_image_list)
print("Inference done")

print("Visualizing predictions")
visualize_predictions(predictions, shortened_test_image_list, kitti_mots_metadata, "output_visualizations")
print("Visualization done")

print("Evaluating predictions")
evaluator = COCOEvaluator("kitti_mots_val_gen", cfg, False, output_dir="./output/")
print('Evaluator created')
val_loader = build_detection_test_loader(cfg, "kitti_mots_val_gen", num_workers=N_WORKERS)
print('Validation loader created')
evaluation_results = inference_on_dataset(predictor.model, val_loader, evaluator)
print("Evaluation done")

# Print the evaluation results
print("Evaluation Results:")
for key, value in evaluation_results.items():
    print(f"{key}: {value}")