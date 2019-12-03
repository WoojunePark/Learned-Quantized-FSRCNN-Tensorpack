# environment related
# DATA_DIR = "/database/wjpark_db/General-100"  # in 'string'

# for server1
# DATA_ZIP_DIR = '/database/wjpark_db/General-100_comp/General-100.zip'   # in 'string'
# LOG_DIR = '/home/wjpark/pycharm_project/tensorpack_study/log_fsrcnn_orgn/'  # in 'string'
# GPU = "0, 1"  # comma separated list of GPU(s) to use. in 'string'
# DATAFLOW_PROC = 2
# LOWRES_DIR = '/home/wjpark/pycharm_project/tensorpack_study/test.zip'
# SROUTPUT_DIR = '/home/wjpark/pycharm_project/tensorpack_study/'

# for server2
TRAIN_DATA_ZIP_DIR = '/database/saehyun/parasite/SR_train_datasets/General-100 train.zip'   # in 'string'
VAL_DATA_ZIP_DIR = '/database/saehyun/parasite/SR_val_datasets/General-100 val.zip'   # in 'string'
LOG_DIR = '/home/saehyun/parasite/pycharm_project/tensorpack_study/log_fsrcnn_orgn/'  # in 'string'
GPU = "0, 1"  # comma separated list of GPU(s) to use. in 'string'
DATAFLOW_PROC = 2
LOWRES_DIR = '/home/saehyun/parasite/pycharm_project/tensorpack_study/test.zip'
SROUTPUT_DIR = '/home/saehyun/parasite/pycharm_project/tensorpack_study/'

# train related
BATCH_SIZE = 4
WEIGHT_DECAY = 5e-4
MAX_EPOCH = 1600
FSRCNN_D = 56
FSRCNN_S = 12
FSRCNN_M = 4
# step for epoch = number of whole dataset / batch
# e.g. 25 = 100 / 4
# 188 epoch (global step : 4700)  4700 = 188*25
# 20000 = 800*25

# FSRCNN_Saafke : 10000 step +

# data related
INPUT_IMAGE_SIZE = 100  # ERROR if INPUT_IMAGE_SIZE % SCALE != 0
NORMALIZE = 1.0  # 1.0 or 255.0
USE_YCBCR = True  # BGR if False
CHANNELS = 1

# hyper params
SCALE = 2  # in 'int'
QA = 4  # quantization activation  in 'int' QW < QA
QW = 8  # quantization weight in 'int'

# parser.add_argument("--dataset",default="data/General-100") V
# parser.add_argument("--savedir",default='saved_models') V

# parser.add_argument("--imgsize",default=100,type=int) V
# parser.add_argument("--layers",default=32,type=int)
# parser.add_argument("--featuresize",default=256,type=int)

# parser.add_argument("--scale",default=2,type=int) V

# parser.add_argument("--batchsize",default=10,type=int) V
# parser.add_argument("--iterations",default=1000,type=int)
# data.load_dataset(args.dataset,args.imgsize)


# down_size = args.imgsize//args.scale
# !!! down_size

# network = EDSR(down_size, args.layers, args.featuresize, args.scale)
# network.set_data_fn(data.get_batch,(args.batchsize, args.imgsize, down_size),data.get_test_set,(args.imgsize,down_size))
# network.train(args.iterations,args.savedir)

