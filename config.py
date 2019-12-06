# train related params
BATCH_SIZE = 1
WEIGHT_DECAY = 5e-4
MAX_EPOCH = 1600
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
QW = 1  # quantization weight in 'int'

# environment related params
TRAIN_DATA_ZIP_DIR = '/database/saehyun/parasite/SR_train_datasets/General-100 train.zip'   # in 'string'
VAL_DATA_ZIP_DIR = '/database/saehyun/parasite/SR_val_datasets/General-100 val.zip'   # in 'string'
LOG_DIR = '/home/saehyun/parasite/pycharm_project/tensorpack_study/log_fsrcnn'+'_qa'+str(QA)+'_qw'+str(QW)+'/'  # in 'string'
GPU = "0, 1"  # comma separated list of GPU(s) to use. in 'string'
DATAFLOW_PROC = 2
LOWRES_DIR = '/home/saehyun/parasite/pycharm_project/tensorpack_study/test/img_005_SRF_2_HR.png'
SROUTPUT_DIR = '/home/saehyun/parasite/pycharm_project/tensorpack_study/sr_output'+'_qa'+str(QA)+'_qw'+str(QW)+'/'  # in 'string'


# python fsrcnn.py --load /home/saehyun/parasite/pycharm_project/tensorpack_study/train_log/fsrcnn_qa32_qw8_ch/checkpoint --apply