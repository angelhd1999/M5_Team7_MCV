import torch
import detectron2
from detectron2.model_zoo import model_zoo
from detectron2.config import get_cfg
import os
import cv2
from glob import glob

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from pycocotools import coco
import numpy as np
import detectron2.utils.comm as comm 
from detectron2.engine import HookBase  # For making hooks
from detectron2.data import (
    build_detection_train_loader,
)  # dataloader is the object that provides the data to the models
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator

from detectron2.engine import DefaultTrainer
import copy
import os
import json
import random
import cv2
import matplotlib.pyplot as plt

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer




KITTI_CORRESPONDENCES = {"Car": 0, "Pedestrian": 1}

def get_kitti_mots_dicts(images_folder, annots_folder, is_train, train_percentage=0.75, image_extension="png"):
    """
    Converts KITTI-MOTS annotations to COCO format and returns a list of dictionaries

    Args:
        images_folder (str): Path to the folder containing images
        annots_folder (str): Path to the folder containing annotations
        is_train (bool): True if creating training data, False otherwise
        train_percentage (float, optional): Percentage of sequences to use for training. Defaults to 0.75.
        image_extension (str, optional): Extension of image files. Defaults to "jpg".

    Returns:
        List[Dict]: A list of dictionaries where each dictionary contains information about an image
    """
    assert os.path.exists(images_folder)
    assert os.path.exists(annots_folder)

    annot_files = sorted(glob(os.path.join(annots_folder, "*.txt")))

    n_train_seqences = int(len(annot_files) * train_percentage)
    train_sequences = annot_files[:n_train_seqences]
    test_sequences = annot_files[n_train_seqences:]

    sequences = train_sequences if is_train else test_sequences

    kitti_mots_annotations = []
    for seq_file in sequences:
        seq_images_path = os.path.join(images_folder, seq_file.split("/")[-1].split(".")[0])
        kitti_mots_annotations += mots_annots_to_coco(seq_images_path, seq_file, image_extension)

    return kitti_mots_annotations


def mots_annots_to_coco(images_path, txt_file, image_extension):
    assert os.path.exists(txt_file)

    # Define the correspondences between class names and class IDs.

    correspondences = KITTI_CORRESPONDENCES

    # Extract the sequence number from the text file name.
    n_seq = int(txt_file.split("/")[-1].split(".")[0])

    mots_annots = []
    with open(txt_file, "r") as f:
        annots = f.readlines()
        annots = [l.split() for l in annots]

        annots = np.array(annots)

        # Iterate over frames in the sequence.
        for frame in np.unique(annots[:, 0].astype("int")):

            # Extract annotations for the current frame.
            frame_lines = annots[annots[:, 0] == str(frame)]
            if frame_lines.size > 0:

                # Extract the image height and width from the first annotation in the frame.
                h, w = int(frame_lines[0][3]), int(frame_lines[0][4])

                f_objs = []
                for a in frame_lines:
                    cat_id = int(a[2]) - 1
                    # Skip annotations that correspond to non-existent classes.
                    if cat_id in correspondences.values():
                        # Extract segmentation information from the annotation.
                        segm = {
                            "counts": a[-1].strip().encode(encoding="UTF-8"),
                            "size": [h, w],
                        }

                        # Convert the segmentation mask to a bounding box.
                        box = coco.maskUtils.toBbox(segm)
                        box[2:] = box[2:] + box[:2]
                        box = box.tolist()

                        # Convert the segmentation mask to a polygon.
                        mask = np.ascontiguousarray(coco.maskUtils.decode(segm))
                        contours, _ = cv2.findContours(
                            mask.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                        )
                        poly = []
                        for contour in contours:
                            contour = contour.flatten().tolist()
                            if len(contour) > 4:
                                poly.append(contour)
                        if len(poly) == 0:
                            continue

                        # Create an annotation dictionary for the current object.
                        annot = {
                            "category_id": cat_id,
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "bbox": box,
                            "segmentation": poly,
                        }
                        f_objs.append(annot)

                # Create a dictionary for the current frame.
                frame_data = {
                    "file_name": os.path.join(
                        images_path, "{:06d}.{}".format(int(a[0]), image_extension)
                    ),
                    "image_id": int(frame + n_seq * 1e6),
                    "height": h,
                    "width": w,
                    "annotations": f_objs,
                }
                mots_annots.append(frame_data)

    return mots_annots

"""
Registering function
"""


def register_kitti_mots_dataset(
    ims_path, annots_path, dataset_names, train_percent=0.75, image_extension="png"
):
    assert isinstance(
        dataset_names, tuple
    ), "dataset names should be a tuple with two strings (for train and test)"

    def kitti_mots_train():
        return get_kitti_mots_dicts(
            ims_path,
            annots_path,
            is_train=True,
            train_percentage=train_percent,
            image_extension=image_extension,
        )

    def kitti_mots_test():
        return get_kitti_mots_dicts(
            ims_path,
            annots_path,
            is_train=False,
            train_percentage=train_percent,
            image_extension=image_extension,
        )

    DatasetCatalog.register(dataset_names[0], kitti_mots_train)
    MetadataCatalog.get(dataset_names[0]).set(
        thing_classes=[k for k, v in KITTI_CORRESPONDENCES.items()]
    )
    DatasetCatalog.register(dataset_names[1], kitti_mots_test)
    MetadataCatalog.get(dataset_names[1]).set(
        thing_classes=[k for k, v in KITTI_CORRESPONDENCES.items()]
    )