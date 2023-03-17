import os
import glob
from PIL import Image
# import cv2
import numpy as np
import pycocotools.mask as cocomask
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog #, build_detection_test_loader
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultPredictor #, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.structures.boxes import BoxMode

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
            # record["sem_seg_file_name"] = inst_path
            dataset_dicts.append(record)

    return dataset_dicts


def setup_config(model_name, model_weights):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATASETS.TEST = ("kitti_mots_testing",)
    return cfg

def run_inference(predictor, image_list):
    predictions = []

    for img_path in image_list:
        img = Image.open(img_path)
        img = np.array(img)
        outputs = predictor(img)
        predictions.append(outputs)

    return predictions


print("Loading KITTI-MOTS dataset")
# Register KITTI-MOTS dataset
for d in ["training", "testing"]:
    print("Registering KITTI-MOTS " + d + " dataset")
    DatasetCatalog.register("kitti_mots_" + d, lambda d=d: get_kitti_mots_dicts("../../../mcv/datasets/KITTI-MOTS", d) )
    MetadataCatalog.get("kitti_mots_" + d).set(thing_classes=["Car", "Pedestrian"])
kitti_mots_metadata = MetadataCatalog.get("kitti_mots_training")

print("Loading models")
faster_rcnn_cfg = setup_config("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl")
faster_rcnn_predictor = DefaultPredictor(faster_rcnn_cfg)

mask_rcnn_cfg = setup_config("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl")
mask_rcnn_predictor = DefaultPredictor(mask_rcnn_cfg)

test_image_list = sorted(glob.glob('../../../mcv/datasets/KITTI-MOTS/testing/image_02/*/*.png'))
print("Running inference on " + str(len(test_image_list)) + " images")

faster_rcnn_predictions = run_inference(faster_rcnn_predictor, test_image_list)
mask_rcnn_predictions = run_inference(mask_rcnn_predictor, test_image_list)

# def evaluate_model(cfg, dataset_name):
#     evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir="./output/")
#     val_loader = build_detection_test_loader(cfg, dataset_name)
#     return inference_on_dataset(predictor.model, val_loader, evaluator)

# def setup_predictor(config_file, model_weights):
#     cfg = get_cfg()
#     cfg.merge_from_file(config_file)
#     cfg.MODEL.WEIGHTS = model_weights
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
#     predictor = DefaultPredictor(cfg)
#     return predictor

# def train_kitti_mots(config_file):
#     cfg = get_cfg()
#     cfg.merge_from_file(config_file)
#     cfg.DATASETS.TRAIN = ("kitti_mots_training",)
#     cfg.DATASETS.TEST = ("kitti_mots_testing",)
#     trainer = DefaultTrainer(cfg)
#     trainer.resume_or_load(resume=False)
#     trainer.train()

# def fine_tune_model(cfg, base_model_name, output_dir, num_classes, num_epochs):
#     cfg.merge_from_file(model_zoo.get_config_file(base_model_name))
#     cfg.DATASETS.TRAIN = ("kitti_mots_training",)
#     cfg.DATASETS.TEST = ("kitti_mots_testing",)
#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_model_name)
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
#     cfg.SOLVER.MAX_ITER = num_epochs
#     cfg.OUTPUT_DIR = output_dir
#     os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#     trainer = DefaultTrainer(cfg)
#     trainer.resume_or_load(resume=False)
#     trainer.train()

# # Faster R-CNN
# print("Evaluating Faster R-CNN")
# faster_rcnn_cfg = setup_config("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", "COCO-Detection/faster_rcnn_R_50_FPN_3x.pkl")

# # Mask R-CNN
# print("Evaluating Mask R-CNN")
# mask_rcnn_cfg = setup_config("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.pkl")

# # Faster R-CNN
# print("Evaluating Faster R-CNN")
# faster_rcnn_predictor = DefaultPredictor(faster_rcnn_cfg)
# faster_rcnn_evaluation_results = evaluate_model(faster_rcnn_cfg, "kitti_mots_testing")

# # Mask R-CNN
# print("Evaluating Mask R-CNN")
# mask_rcnn_predictor = DefaultPredictor(mask_rcnn_cfg)
# mask_rcnn_evaluation_results = evaluate_model(mask_rcnn_cfg, "kitti_mots_testing")


# # Load an image from the KITTI-MOTS dataset
# image_path = "../../../mcv/datasets/KITTI-MOTS/training/image_02/0000/000000.png"
# im = cv2.imread(image_path)

# # Run inference
# outputs = predictor(im)
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# out_image = out.get_image()[:, :, ::-1]
# cv2.imwrite("inference_output.png", out_image)