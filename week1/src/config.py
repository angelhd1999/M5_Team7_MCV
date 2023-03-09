#data configuration feeding into the model
INPUT_WIDTH = 224
INPUT_HEIGHT = 224
INPUT_CHANNEL = 3
BATCH_SIZE = 16
NUM_WORKERS = 1

#location of the dataset, save model, etc.
TRAIN_DATA_LOC = r'./MIT_split/train'
TEST_DATA_LOC = r'./MIT_split/test'
ANNOT_LOC = r'./data'
MODEL_SAVE_LOC = r'./checkpoint'
REPORT_SAVE_LOC = r'./report'