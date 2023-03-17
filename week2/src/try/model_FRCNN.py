import os
from glob import glob
import random
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
import detectron2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_train_loader
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultPredictor, DefaultTrainer, HookBase
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.model_zoo import model_zoo
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from pycocotools import coco

from kitti_mots import register_kitti_mots_dataset
from kitti_mots import get_kitti_mots_dicts
from detectron2_helpers import ValidationLoss
from detectron2_helpers import plot_losses
from detectron2_helpers import show_results

KITTI_CORRESPONDENCES = {"Car": 0, "Pedestrian": 1}

'''Register the Kitti-Mots dataset'''
register_kitti_mots_dataset(
    "/Users/eduard.hogea/Documents/Facultate/Erasmus/UAB_sem2/M5/M5-Visual-Recognition/datasets/KITTI_MOTS/data_tracking_image_2/training/image_02",
    annots_path="/Users/eduard.hogea/Documents/Facultate/Erasmus/UAB_sem2/M5/M5-Visual-Recognition/datasets/KITTI_MOTS/instances_txt",
    dataset_names=("k", "t"),
    image_extension="png",
)

'''Updating the config file'''
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.DATASETS.TRAIN = "k"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.DATASETS.TEST = "t"
cfg.DATALOADER.NUM_WORKERS = 0
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.0002 * 2 * 1.4 / 16
cfg.SOLVER.MAX_ITER = 200
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"
cfg.OUTPUT_DIR = "output_week2"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

'''Training part using hooks'''
cfg.MODEL.WEIGHTS = "/Users/eduard.hogea/Documents/Facultate/Erasmus/UAB_sem2/M5/M5-Visual-Recognition/output_week2/mymodelfull.pth"
trainer = DefaultTrainer(cfg)

val_loss = ValidationLoss(cfg)
trainer.register_hooks([val_loss])

print(trainer._hooks)
trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
print(trainer._hooks)

trainer.resume_or_load(resume=True)
trainer.train()

torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "mymodelfull.pth"))
checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)

"""
Visualization
"""
plot_losses(cfg)

evaluator = COCOEvaluator("t", cfg, False, output_dir="output_week2") #evaluate the model with COCO metrics
results_coco = trainer.test(cfg, trainer.model, evaluators=[evaluator]) # !! it evaliuates on the cfg test data
with open("output_week2/evaluate.json", "w") as outfile:
    json.dump(results_coco, outfile)

predictor = DefaultPredictor(cfg)
predictor.model.load_state_dict(trainer.model.state_dict())

dataset_dicts = get_kitti_mots_dicts(
    "/Users/eduard.hogea/Documents/Facultate/Erasmus/UAB_sem2/M5/M5-Visual-Recognition/datasets/KITTI_MOTS/data_tracking_image_2/training/image_02",
    "/Users/eduard.hogea/Documents/Facultate/Erasmus/UAB_sem2/M5/M5-Visual-Recognition/datasets/KITTI_MOTS/instances_txt",
    is_train=False,
    image_extension="png",
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
show_results(cfg, dataset_dicts, predictor, samples=10)
