import os
import glob
from PIL import Image
import cv2
import numpy as np
import torch
import pycocotools.mask as cocomask
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog #, build_detection_test_loader
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultPredictor #, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.structures.boxes import BoxMode
from detectron2.utils.visualizer import Visualizer

DATASET_PATH = '../../../mcv/datasets/KITTI-MOTS'
MODEL = 'FasterRCNN' # 'MaskRCNN' or 'FasterRCNN'
N_IMAGES_TEST = 10

print("Starting script")

def get_kitti_mots_dicts(data_path, mode):
    # assert mode in ['training', 'testing'], "mode should be 'training' or 'testing'"
    
    image_path = os.path.join(data_path, mode, 'image_02')
    instance_path = os.path.join(data_path, 'instances')
    instance_txt_path = os.path.join(data_path, 'instances_txt')
    
    dataset_dicts = []

    for seq in sorted(os.listdir(image_path)):
        images = sorted(glob.glob(os.path.join(image_path, seq, '*.png')))
        instances = sorted(glob.glob(os.path.join(instance_path, seq, '*.png')))
        instance_txt = os.path.join(instance_txt_path, seq + '.txt')
        
        with open(instance_txt, 'r') as f:
            instance_data = f.readlines()

        for idx, (img_path, inst_path) in enumerate(zip(images, instances)):
            # Log progress
            if idx % 100 == 0:
                print("Processing sequence " + seq + ", image " + str(idx) + "/" + str(len(images)))
            record = {}
            img = Image.open(img_path)
            width, height = img.size

            record["file_name"] = img_path
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
            
            # Parse instance data
            objs = []
            for line in instance_data:
                values = line.strip().split(' ')
                frame_id, class_instance_ids, class_id_re, in_height, in_width, rle_encoding = values
                
                frame_id = int(frame_id)

                # Parse class and instance id
                class_id = int(class_instance_ids) // 1000
                class_id_re = int(class_id_re) # ? It's redundant, but we'll keep it for now
                instance_id = int(class_instance_ids) % 1000
                
                # Decode RLE mask encoding
                rle = {'counts': rle_encoding, 'size': [int(in_height), int(in_width)]}
                binary_mask = cocomask.decode(rle)

                if class_id_re > 2 or np.sum(binary_mask) == 0:
                    continue
                    
                print("Found object with class id " + str(class_id) + " and instance id " + str(instance_id) + " in frame " + str(frame_id))
                # Compute the bounding box from the mask
                y_indices, x_indices = np.where(binary_mask)
                bbox = [np.min(x_indices), np.min(y_indices), np.max(x_indices) - np.min(x_indices), np.max(y_indices) - np.min(y_indices)]
                
                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": class_id,
                    "instance_id": instance_id,
                    "segmentation": rle,
                }
                objs.append(obj)

            record["annotations"] = objs
            # record["sem_seg_file_name"] = inst_path # ? Not sure if we need this
            dataset_dicts.append(record)

    return dataset_dicts


def setup_config(model_name):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Set threshold for this model
    cfg.DATASETS.TEST = ("kitti_mots_testing",)
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
    class_mapping = {  # Mapping COCO class indices to KITTI-MOTS class indices
        3: 0, # COCO class 3 is "car" so we map it to KITTI-MOTS class 0
        1: 1 # COCO class 1 is "person" so we map it to KITTI-MOTS class 1
    } 
    ignore_class = 2

    # Add all other COCO classes to the mapping dictionary as an "Ignore" class
    for i in range(80):
        if i not in class_mapping:
            class_mapping[i] = ignore_class

    for i, (pred, img_path) in enumerate(zip(predictions, image_list)):
        print(f"Visualizing prediction {i + 1}/{len(image_list)}: {img_path}")
        img = cv2.imread(img_path)
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        
        instances = pred["instances"].to("cpu")
        instances.pred_classes = torch.tensor([class_mapping[c.item()] for c in instances.pred_classes])  # Apply the class mapping
        filtered_instances = instances[instances.pred_classes != ignore_class]  # Filter out the "ignore" class
        vis_output = visualizer.draw_instance_predictions(filtered_instances)
        
        vis_img = vis_output.get_image()[:, :, ::-1]
        output_file = os.path.join(output_dir, os.path.dirname(img_path).split('/')[-1] + '_' + os.path.basename(img_path))
        cv2.imwrite(output_file, vis_img)

print("Loading KITTI-MOTS dataset")
# Register KITTI-MOTS dataset
for d in ["training", "testing"]:
    print("Registering KITTI-MOTS " + d + " dataset")
    DatasetCatalog.register("kitti_mots_" + d, lambda d=d: get_kitti_mots_dicts(DATASET_PATH, d) )
    MetadataCatalog.get("kitti_mots_" + d).set(thing_classes=["Car", "Pedestrian", "Ignore"])

# Get the metadata for the KITTI-MOTS training dataset, used for visualizing predictions
kitti_mots_metadata = MetadataCatalog.get("kitti_mots_training")

print(f"Loading models and weights from {MODEL} model") 
if MODEL == 'FasterRCNN':
    cfg = setup_config("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
elif MODEL == 'MaskRCNN':
    cfg = setup_config("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
else:
    raise ValueError("MODEL should be 'MaskRCNN' or 'FasterRCNN'")

predictor = DefaultPredictor(cfg)

print("Loading test images")
test_image_list = sorted(glob.glob(DATASET_PATH + '/testing/image_02/*/*.png'))
shortened_test_image_list = test_image_list[:N_IMAGES_TEST]

print("Running inference on " + str(len(shortened_test_image_list)) + " images")
predictions = run_inference(predictor, shortened_test_image_list)
print("Inference done")

print("Visualizing predictions")
visualize_predictions(predictions, shortened_test_image_list, kitti_mots_metadata, "output_visualizations")
print("Visualization done")