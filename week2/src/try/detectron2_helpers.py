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



class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()  # takes init from HookBase
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self._loader = iter(
            build_detection_train_loader(self.cfg)
        )  # builds the dataloader from the provided cfg
        self.best_loss = float("inf")  # Current best loss, initially infinite
        self.weights = None  # Current best weights, initially none
        self.i = 0  # Something to use for counting the steps

    def after_step(self):  # after each step

        if self.trainer.iter >= 0:
            print(
                f"----- Iteration num. {self.trainer.iter} -----"
            )  # print the current iteration if it's divisible by 100

        data = next(self._loader)  # load the next piece of data from the dataloader

        with torch.no_grad():  # disables gradient calculation; we don't need it here because we're not training, just calculating the val loss
            loss_dict = self.trainer.model(data)  # more about it in the next section

            losses = sum(loss_dict.values())  #
            assert torch.isfinite(losses).all(), loss_dict
            loss_dict_reduced = {
                "val_" + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            if comm.is_main_process():
                self.trainer.storage.put_scalars(
                    total_val_loss=losses_reduced, **loss_dict_reduced
                )  # puts these metrics into the storage (where detectron2 logs metrics)

                # save best weights
                if losses_reduced < self.best_loss:  # if current loss is lower
                    self.best_loss = losses_reduced  # saving the best loss
                    self.weights = copy.deepcopy(
                        self.trainer.model.state_dict()
                    )  # saving the best weights

def plot_losses(cfg):

    val_loss = []
    train_loss = []
    for line in open(os.path.join(cfg.OUTPUT_DIR, "metrics.json"), "r"):
        if (
            "total_val_loss" in json.loads(line).keys()
            and "total_loss" in json.loads(line).keys()
        ):
            val_loss.append(json.loads(line)["total_val_loss"])
            train_loss.append(json.loads(line)["total_loss"])

    plt.plot(val_loss, label="Validation Loss")
    plt.plot(train_loss, label="Training Loss")
    plt.legend()
    plt.show()


def show_results(cfg, dataset_dicts, predictor, samples=10):

    for data in random.sample(dataset_dicts, samples):
        im = cv2.imread(data["file_name"])
        outputs = predictor(im)
        # print(outputs)

        # outputs["instances"] = outputs["instances"][torch.where(outputs["instances"].pred_classes < 2)]
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("t"), scale=0.5)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("Frame", v.get_image()[:, :, ::-1])
        cv2.waitKey(0)