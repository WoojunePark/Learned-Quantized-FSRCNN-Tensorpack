import argparse
import numpy as np
import os
import cv2
import six

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer

from tensorpack import *
from tensorpack.dataflow.serialize import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import *
from tensorpack.utils import logger
from tensorpack.utils.gpu import get_num_gpu

import config
from load_data import get_data
from learned_quantization import *

SHAPE_LR = 100
CHANNELS = 3


class Model(ModelDesc):
    def __init__(self, height=SHAPE_LR, width=SHAPE_LR, qw=1, qa=1):
        super(Model, self).__init__()
        self.height = height
        self.width = width

        self.qw = qw
        self.qa = qa

        self.name = "FSRCNN"

        # from primitive FSRCNN
        # Different model layer counts and filter sizes for FSRCNN vs FSRCNN-s (fast), (d, s, m) in paper
        model_params = [32, 0, 4, 1]
        self.model_params = model_params

        self.radius = config.radius
        self.padding = config.padding
        self.images = config.images
        self.image_size = config.image_size - self.padding
        self.label_size = config.label_size

    def inputs(self):
        return [tf.TensorSpec((None, self.height*1,self.width*1, CHANNELS), tf.float32, 'input_lr'),
                tf.TensorSpec((None, self.height*4,self.width*4, CHANNELS), tf.float32, 'input_hr')
                ]

    def build_graph(self, input_lr, input_hr):
        input_lr, input_hr = input_lr / 255.0, input_hr / 255.0
        input_bicubic = tf.image.resize_bicubic(
            input_lr, [self.height*4, self.width*4], align_corners=True, name='baseline_bicubic')

        assert tf.test.is_gpu_available()
        # image = tf.transpose(image, [0, 3, 1, 2])

        with argscope([Conv2DQuant, MaxPooling, BatchNorm], data_format="NCHW"), \
            argscope(Conv2DQuant, nl=tf.identity,
                     use_bias=False,
                     kernel_shape=3,
                     W_init=variance_scaling_initializer(mode='FAN_IN'),
                     nbit=self.qw,
                     is_quant=True if self.qw > 0 else False):

            # feature extraction : 5x5 convolutions.
            layer = Conv2DQuant('conv0', image, 128, nl=BNReLU, is_quant=False)

            # shrinking : Reduction in feature maps.
            layer = "..."

            # non-linear mappping : Multiple layers are applied 3x3.
            layer = "..."

            # expanding : The feature map is now increased by 1x1 convolutions.
            layer = "..."

            # transposed conv : High resolution image is reconstructed using 9x9 filter.
            layer = "..."

            # layer related lines...

            # cost related lines...

            return self.cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.02, trainable=False)
        # choose which optimizer?
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load previous model')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU

    logger.set_logger_dir(
        os.path.join('train_log', config.LOG_DIR))

    dataset_train = QueueInput(get_data(config.DATA_ZIP_DIR, 'train'))
    dataset_test = QueueInput(get_data(config.DATA_ZIP_DIR, 'test'))

    train_config = TrainConfig(
        model=Model(qw=config.QW, qa=config.QA),
        data=dataset_train,
        callbacks=[
            ModelSaver(keep_checkpoint_every_n_hours=2),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong_vector')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.02), (80, 0.002), (160, 0.0002), (300, 0.00002)])
        ],
        max_epoch=config.MAX_EPOCH,
        nr_tower=max(get_num_gpu(), 1),
        session_init=SaverRestore(args.load) if args.load else None
    )
    num_gpu = max(get_num_gpu(), 1)
    launch_train_with_config(train_config, SyncMultiGPUTrainerParameterServer(num_gpu))
