# environment related
DATA_DIR = "/database/wjpark_db/General-100"  # in 'string'
LOG_DIR = ""  # in 'string'
GPU = "0, 3"  # comma separated list of GPU(s) to use. in 'string'

# train related
BATCH_SIZE = 100
WEIGHT_DECAY = 5e-4
MAX_EPOCH = 200

# data related
INPUT_IMAGE_SIZE = 50
# ERROR if INPUT_IMAGE_SIZE % SCALE != 0

# hyper params
SCALE = 2  # in 'int'
QW = 1  # quantization weight in 'int'
QA = 1  # quantization activation  in 'int'


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

