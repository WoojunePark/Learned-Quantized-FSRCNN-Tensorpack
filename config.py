# train related params
BATCH_SIZE = 64
WEIGHT_DECAY = 5e-4
MAX_EPOCH = 900000
FSRCNN_D = 56
FSRCNN_S = 12
FSRCNN_M = 4

ORIGINAL_FSRCNN = False  # if True, QW, QA are ignored.

# data related params
INPUT_IMAGE_SIZE = 100  # ERROR if INPUT_IMAGE_SIZE % SCALE != 0
NORMALIZE = 1.0  # 1.0 or 255.0
USE_YCBCR = True  # BGR if False
CHANNELS = 1

# hyper params
SCALE = 2  # in 'int'
QA = 0  # quantization activation  in 'int' QW < QA
QW = 4  # quantization weight in 'int'

# environment related params
TRAIN_DATA_ZIP_DIR = '/database/wjpark/SR_train_datasets/T91+General-100_full.zip'   # in 'string'
VAL_DATA_ZIP_DIR = '/database/wjpark/SR_val_datasets/Set5+14_full.zip'   # in 'string'
LOG_DIR = '/home/wjpark/pycharm_project/tensorpack_study/log_fsrcnn'+'_qa'+str(QA)+'_qw'+str(QW)+'/'  # in 'string'
GPU = "0"  # comma separated list of GPU(s) to use. in 'string'
DATAFLOW_PROC = 2
LOWRES_DIR = '/home/wjpark/pycharm_project/tensorpack_study/apply_dataset/Set14'
# LOWRES_DIR_ZIP = '/home/wjpark/pycharm_project/tensorpack_study/apply_dataset/Set14.zip'
# SROUTPUT_DIR = '/home/wjpark/pycharm_project/tensorpack_study/sr_output'+'_qa'+str(QA)+'_qw'+str(QW)+'/'  # in 'string'
SROUTPUT_DIR = '/home/wjpark/pycharm_project/tensorpack_study/sr_output/'  # in 'string'


# python fsrcnn.py --load /home/wjpark/pycharm_project/tensorpack_study/train_log/fsrcnn_orgn/checkpoint --apply
