# M5_Team7_MCV
Team 7 of the M5 at the Master in Computer Vision.

## Team members
- **Àngel Herrero Díaz** - herreroangel.pro@gmail.com
- **Marcos Frías Nestares** - marcos.frias.00@gmail.com
- **Adriá Subirana Pérez** - adria.subi@gmail.com
- **Ayan Banerjee** - ab2141@cse.jgec.ac.in

## Installation
 1. Go to the desired week:
```
:: Where N is the week number
cd weekN
```
 2. (Optional) Create an environment.
 3. Install the requirements at the folder:
```
pip install -r requirements.txt
```

## Execution Week 1
```
python run.py
```
## Execution Week 2
This script is designed to fine-tune and evaluate a model on the KITTI-MOTS dataset. It supports both Faster R-CNN and Mask R-CNN models. Usage:  
```sh
python script.py [--dataset_path PATH] [--model MODEL_NAME] [--finetuning] [--images_test_start START_INDEX] [--n_images_test N_IMAGES] [--n_workers N_WORKERS]
```
### Parameters

- --dataset_path PATH``` (default: ```../../../KITTI-MOTS```): Path to the KITTI-MOTS dataset.
- --model MODEL_NAME``` (default: ```FasterRCNN```): Model to use for training and evaluation. Choose either ```'MaskRCNN'``` or ```'FasterRCNN'```.
- --finetuning```: Enable fine-tuning of the model. If this flag is provided, the script will fine-tune the model on the KITTI-MOTS dataset.
- --images_test_start START_INDEX``` (default: ```0```): Start index for test images to use for inference and visualization.
- --n_images_test N_IMAGES``` (default: ```10```): Number of test images to use for inference and visualization.
- --n_workers N_WORKERS``` (default: ```8```): Number of workers to use for data loading.

### Example
```sh
python script.py --dataset_path /path/to/kitti-mots --model FasterRCNN --finetuning --images_test_start 5 --n_images_test 20 --n_workers 4
```
This command will run the script with the Faster R-CNN model, enable fine-tuning, start inference and visualization from the 5th test image, process 20 test images, and use 4 workers for data loading.

## Slides
- [Deliverables folder with the slides](https://drive.google.com/drive/folders/1u2li3fMPq72JS9kjdGnuZzbt4MwzZuf5?usp=sharing)

## Overleaf
- [Overleaf link](https://www.overleaf.com/read/kfmrcrgyvrft)
