MODEL_MAP  = {
    'LW'  : 'mobilenet',
    'MW'  : 'resnet50',
    'HW'  : 'resnet150'
}

DEFAULT_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE  = 224
DEFAULT_OPTIMISER = 'radam'
DEFAULT_AUGMENTATIONS = {
    'do_flip' : True,
    'flip_horizontly' : True,
    'zoom' : 0.2,
}

DATASET_COLLECTION = 'datasets'
IMAGE_DATASET_COLLECTION = 'img_dataset'
DEFAULT_METRICS = ['accuracy']
GPU_QUEUE_URL = 'tcp://localhost:5551'
DEFAULT_LEARNING_RATE = 0.001
WORKSTATION_COLLECTION = 'workstations'
PARTS_COLLECTION = 'parts'
SHIFT_COLLECTION = 'shift'
INSPECTION_DATA_COLLECTION = 'inspection_data'

# just for testing -annotation tool
#IMAGE_DATASET_SAVE_PATH = '/home/nitin/IMG_dATASET/'
#PARTS_COLLECTION_TMP = 'parts_tmp'    #main livis 

import os
this_pth = os.getcwd()
full_ext_pth = os.path.join(this_pth,"annotate","extract")
IMAGE_DATASET_SAVE_PATH = full_ext_pth


