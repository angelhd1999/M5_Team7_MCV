import os
import glob
import json
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

DATASET_PATH = '../../../KITTI-MOTS'
MODEL = 'FasterRCNN' # 'MaskRCNN' or 'FasterRCNN'
FINETUNING = True
N_IMAGES_TEST = 10
N_WORKERS = 8

class_mapping_k_to_c = {  # Mapping KITTI-MOTS class indices to COCO class indices
    1: 2, # KITTI-MOTS class 1 is "car" when read, so we map it to COCO class 2
    2: 1 # KITTI-MOTS class 2 is "person" when read, so we map it to COCO class 1
}

class_mapping_c_to_k = {  # Mapping COCO class indices to KITTI-MOTS class indices
    2: 0, # COCO class 2 is "car" so we map it to KITTI-MOTS class 0 (defined at thing_classes)
    1: 1 # COCO class 1 is "person" so we map it to KITTI-MOTS class 1 (defined at thing_classes)
} 
ignore_class = 2

# Add all other COCO classes to the mapping dictionary as an "Ignore" class
for i in range(80):
    if i not in class_mapping_c_to_k:
        class_mapping_c_to_k[i] = ignore_class

print("Starting script")

# Enble CUDA
print("Enabling CUDA")
torch.cuda.set_device(0)

def get_kitti_mots_dicts(data_path, mode, model):
    # assert mode in ['training', 'testing'], "mode should be 'training' or 'testing'"
    
    image_path = os.path.join(data_path, mode, 'image_02')
    instance_path = os.path.join(data_path, 'instances')
    instance_txt_path = os.path.join(data_path, 'instances_txt' + '_' + mode)

    dataset_dicts = []

    for seq in sorted(os.listdir(image_path)):
        images = sorted(glob.glob(os.path.join(image_path, seq, '*.png')))
        instances = sorted(glob.glob(os.path.join(instance_path, seq, '*.png')))
        instance_txt = os.path.join(instance_txt_path, seq + '.txt')
        
        # Log progress
        print("Loading " + str(len(images)) + " images from sequence " + seq)

        with open(instance_txt, 'r') as f:
            instance_data = f.readlines()

        for idx, (img_path, inst_path) in enumerate(zip(images, instances)):
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
                in_height = int(in_height)
                in_width = int(in_width)
                # Parse class and instance id
                class_id = int(class_instance_ids) // 1000
                class_id_re = int(class_id_re) # ? It's redundant, but we'll keep it for now
                instance_id = int(class_instance_ids) % 1000
                
                # Decode RLE mask encoding
                rle = {'counts': rle_encoding, 'size': [in_height, in_width]}
                binary_mask = cocomask.decode(rle)

                if class_id_re > 2 or np.sum(binary_mask) == 0:
                    continue
                    
                # print("Found object with class id " + str(class_id) + " and instance id " + str(instance_id) + " in frame " + str(frame_id))
                # Compute the bounding box from the mask
                y_indices, x_indices = np.where(binary_mask == 1)
                bbox = [int(np.min(x_indices)), int(np.min(y_indices)), int(np.max(x_indices) - np.min(x_indices)), int(np.max(y_indices) - np.min(y_indices))]
                
                # Map class id to COCO class id
                class_id = class_mapping_k_to_c[class_id]

                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": class_id,
                    "instance_id": instance_id,
                }
                objs.append(obj)

                if (model == 'MaskRCNN'):
                    shape = [(x_indices, y_indices) for x_indices, y_indices in zip(x_indices, y_indices)]
                    shape = [s for x in shape for s in x]
                    obj["segmentation"] = shape

            record["annotations"] = objs
            # record["sem_seg_file_name"] = inst_path # ? Not sure if we need this
            dataset_dicts.append(record)

    return dataset_dicts


def setup_config_predef(model_name):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Set threshold for this model
    cfg.DATASETS.TEST = ("kitti_mots_testing",)
    cfg.MODEL.FP16_ENABLED = True # Enable mixed precision training for faster inference
    cfg.MODEL.DEVICE='cuda'
    return cfg

def setup_config_finetuning(model_name, train_dataset_name, val_dataset_name):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.VAL = (val_dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.FP16_ENABLED = True  # Enable mixed precision training for faster inference
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # You can adjust this value depending on your GPU memory
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Number of classes in KITTI-MOTS dataset (excluding the ignore class)
    cfg.MODEL.DEVICE='cuda'

    # Set up the training parameters
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 1000  # You can adjust the number of iterations based on your needs
    cfg.SOLVER.STEPS = []  # Do not decay learning rate
    cfg.SOLVER.GAMMA = 0.05
    cfg.SOLVER.CHECKPOINT_PERIOD = 500  # Save a checkpoint every 500 iterations
    cfg.TEST.EVAL_PERIOD = 500  # Evaluate the model every 500 iterations


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
    MetadataCatalog.get("kitti_mots_" + d).set(thing_classes=["Car", "Pedestrian", "Ignore"])

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


if FINETUNING:
    # Create a DefaultTrainer
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # Train the model
    trainer.train()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

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

print("Evaluating predictions")
evaluator = COCOEvaluator("kitti_mots_val_gen", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "kitti_mots_val_gen", num_workers=N_WORKERS)
evaluation_results = inference_on_dataset(predictor.model, val_loader, evaluator)
print("Evaluation done")

# Print the evaluation results
print("Evaluation Results:")
for key, value in evaluation_results.items():
    print(f"{key}: {value}")