# Import required packages
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt

# Run a pre-trained detectron2 model
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

im = cv2.imread("/Users/advaitdixit/Downloads/data_tracking_image_2/testing/image_02/0006/000026.png")
print(im.shape)
plt.figure(figsize=(15, 7.5))
plt.imshow(im[..., ::-1])

outputs = predictor(im[..., ::-1])
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imwrite("output.png", out.get_image()[:, :, ::-1])
plt.figure(figsize=(20, 10))
plt.imshow(out.get_image()[..., ::-1][..., ::-1])
