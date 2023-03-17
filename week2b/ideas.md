Question:
I need to use detectron 2 and pytorch and perform the next steps:
- Run inference with pre-trained Faster R-CNN (detection) and Mask R-CNN (detection and segmentation) on KITTI-MOTS dataset.
- Evaluate pre-trained Faster R-CNN (detection) and Mask R-CNN (detection and segmentation) on KITTI-MOTS dataset.
- Fine-tune Faster R-CNN and Mask R-CNN on KITTI-MOTS
KITTI-MOTS dataset (video dataset for cars and pedestrian classification) is in the following format:
In the folder at "path_to_data/KITTI-MOTS" there are 4 folders:
- /instances: Folders from 0000 to 0020, inside every folder there are pngs of the masks of each frame.
- /instances_txt: Txt. files from 0000 to 0020, contains codified information of each frame.
- /testing: Inside folder /image_02 there are folders from 0000 to 0020 that contain frames (images) for testing.
- /training:  Inside folder /image_02 there are folders from 0000 to 0020 that contain frames (images) for training.
As reference, this code works to simply use detectron to classify images and specifically ballons at COCO dataset:
``` python
DATA_PATH = "./fake_data"

import torch, detectron2
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print(torch.__version__)
print("detectron2:", detectron2.__version__)

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Testing cv2 library
im = cv2.imread(DATA_PATH + "/input.jpg")
cv2.imwrite(DATA_PATH + "/testread.jpg", im)

# Then, we create a detectron2 config and a detectron2 DefaultPredictor to run inference on this image.
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
out_image = out.get_image()[:, :, ::-1]
cv2.imwrite(DATA_PATH + "/testout.jpg", out_image)

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

from detectron2.structures import BoxMode

def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
balloon_metadata = MetadataCatalog.get("balloon_train")

dataset_dicts = get_balloon_dicts("balloon_data/balloon/train")
for idx, d in enumerate(random.sample(dataset_dicts, 3)):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    out_image = out.get_image()[:, :, ::-1]
    cv2.imwrite(DATA_PATH + f"/testout{idx}.jpg", out_image)
```

Response:
In order to adapt this code to your needs, you should first change the dataset registration part to work with the KITTI-MOTS dataset. You'll need to create a function to parse the KITTI-MOTS dataset and register it in Detectron2. Here's how you can do that:

1. Create a function to parse the KITTI-MOTS dataset:
``` python
import os
import glob
from PIL import Image

def get_kitti_mots_dicts(data_path, mode):
    assert mode in ['training', 'testing'], "mode should be 'training' or 'testing'"
    
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
                frame_id, class_id, bbox = line.strip().split('\t')
                frame_id = int(frame_id)
                
                if frame_id != idx:
                    continue
                
                x_min, y_min, x_max, y_max = [float(x) for x in bbox.split(',')]
                
                obj = {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": int(class_id),
                }
                objs.append(obj)
            
            record["annotations"] = objs
            dataset_dicts.append(record)

    return dataset_dicts
```
2. Register KITTI-MOTS dataset:
``` python
for d in ["training", "testing"]:
    DatasetCatalog.register("kitti_mots_" + d, lambda d=d: get_kitti_mots_dicts("path_to_data/KITTI-MOTS", d))
    MetadataCatalog.get("kitti_mots_" + d).set(thing_classes=["Car", "Pedestrian"])
kitti_mots_metadata = MetadataCatalog.get("kitti_mots_training")
```
Now you can run inference and evaluation with the pre-trained Faster R-CNN and Mask R-CNN on the KITTI-MOTS dataset.
Inference:
``` python
# Load an image from the KITTI-MOTS dataset
image_path = "path_to_data/KITTI-MOTS/training/image_02/0000/000000.png"
im = cv2.imread(image_path)

# Run inference
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
out_image = out.get_image()[:, :, ::-1]
cv2.imwrite("inference_output.png", out_image)
```
Evaluation:
``` python
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detect
```