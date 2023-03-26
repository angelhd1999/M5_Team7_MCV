import cv2

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from configs import OUTPUT_DIR


def main():
    dataset_path = Path(OUTPUT_DIR)
    out_path = Path(OUTPUT_DIR)
    out_path = out_path / "segmentation"
    print(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    MODEL = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = DefaultPredictor(cfg)
    print(dataset_path)
    for img_path in dataset_path.glob("*.png"):
        im = cv2.imread(str(img_path))

        # Inference
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        print("out",out)
        print("img name",img_path.parts[-1])

        cv2.imwrite(
            str(out_path / img_path.parts[-1]),
            out.get_image()[:, :, ::-1],
            [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
        )

    print("PROCESS FINISHED")


if __name__ == "__main__":
    main()
