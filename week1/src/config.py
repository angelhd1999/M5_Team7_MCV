#data configuration feeding into the model
INPUT_WIDTH = 224
INPUT_HEIGHT = 224
INPUT_CHANNEL = 3
BATCH_SIZE = 16
NUM_WORKERS = 1

#location of the dataset, save model, etc.
DATASETS_DIR = r'../../../mcv/datasets/'
TRAIN_DATA_LOC = DATASETS_DIR + r'MIT_split/train'
TEST_DATA_LOC = DATASETS_DIR + r'MIT_split/test'
ANNOT_LOC = r'../data'
MODEL_SAVE_LOC = r'../checkpoints'
REPORT_SAVE_LOC = r'../report'

