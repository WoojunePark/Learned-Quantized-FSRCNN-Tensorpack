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

from PIL import Image

import config
from load_data import get_data
from learned_quantization import *

SHAPE_LR = 100
CHANNELS = 3


class Model(ModelDesc):
    def __init__(self, d, s, m, height=SHAPE_LR, width=SHAPE_LR, qw=1, qa=1, ):
        super(Model, self).__init__()
        self.height = height
        self.width = width

        self.qw = qw
        self.qa = qa

        self.name = "FSRCNN"

        self.d = d
        self.s = s
        self.m = m

        # from primitive FSRCNN
        # Different model layer counts and filter sizes for FSRCNN vs FSRCNN-s (fast), (d, s, m) in paper
        # model_params = [32, 0, 4, 1]
        # self.model_params = model_params

        # self.radius = config.radius
        # self.padding = config.padding
        # self.images = config.images
        # self.image_size = config.image_size - self.padding
        # self.label_size = config.label_size

    def inputs(self):
        return [tf.TensorSpec((None, self.height*1,self.width*1, CHANNELS), tf.float32, 'input_lr'),
                # tf.TensorSpec((None, self.height*4,self.width*4, CHANNELS), tf.float32, 'input_hr')
                ]

    def build_graph(self, input_x):
        input_x = input_x / 255.0

        d = self.d
        s = self.s
        m = self.m
        input_bicubic = tf.image.resize(
            input_x, [50, 50], method=tf.image.ResizeMethod.BICUBIC,
            name='bicubic_baseline')

        assert tf.test.is_gpu_available()
        # input_lr = tf.transpose(image, [0, 3, 1, 2])

        # (1(=possibly number of steams?..) , Batch_size, image_h, image_w, 3 ch.(RGB))

        # tf.nn.con2d  default=NHWC
        # input = [batch, in_h, in_w, in_c]
        # filter = [filter_h, filter_w, in_c, out_c]

        # flattens the filter to [filter_h * filter_w * in_c, out_c]

        # [batch, out_h, out_w, filter_h * filter_w * in_c]
        # 'NHWC' or 'NCHW'. Defaults to 'NHWC'.

        with argscope([Conv2DQuant, MaxPooling, BatchNorm], data_format="NHWC"), \
            argscope(Conv2DQuant,
                     padding='same',
                     kernel_shape=1,
                     stride=1,
                     W_init=variance_scaling_initializer(mode='FAN_IN'),
                     # b_init=tf.constant_initializer(value=0.0),
                     nl=PReLU_4Q,
                     use_bias=False,
                     is_quant=True if self.qw > 0 else False,
                     nbit=self.qw):

            channels = 1
            PS = channels * 4  # for sub-pixel, PS = Phase Shift
            # bias_initializer = tf.constant_initializer(value=0.0)
            # input_bicubic = tf.transpose(input_bicubic, [0, 3, 1, 2])

            # -- Model architecture --
            # feature extraction : 5x5 convolutions. (Non-Quantized)
            print('i_bi : ', input_bicubic)
            layer = Conv2DQuant('1_Fe_Ex', input_bicubic, d, kernel_shape=5, is_quant=False)
            print('__fd : ', layer)

            # shrinking : Reduction in feature maps.
            layer = Conv2DQuant('2_Shrnk', layer, s)
            print('__sh : ', layer)
            if self.qa > 0:
                layer = QuantizedActiv('2_Shrnk_QA', layer, self.qa)
                print('q_sh : ', layer)

            # non-linear mappping : Multiple layers are applied 3x3.
            for i in range(0, m):
                layer = Conv2DQuant('3_Nl_Ma'+str(i), layer, s, kernel_shape=3)
                if self.qa > 0:
                    layer = QuantizedActiv('3_Nl_Ma_QA'+str(i), layer, self.qa)
                    print('q_nl', str(i), ": ", layer)

            # expanding : The feature map is now increased by 1x1 convolutions.
            layer = Conv2DQuant('4_Expan', layer, d)
            if self.qa > 0:
                layer = QuantizedActiv('4_Expan_QA', layer, self.qa)
                print('q_ex : ', layer)

            # 1) transposed conv : High resolution image is reconstructed using 9x9 filter.
            # layer = tf.nn.conv2d_transpose('tr', input=layer, filters=3,
            #                                output_shape=[config.BATCH_SIZE, self.height*2, self.width*2, 1],
            #                                strides=[1, 2, 2, 1],
            #                                padding='SAME')

            # 2) sub-pixel
            layer = Conv2DQuant('5_Su_Px', layer, 12)
            if self.qa > 0:
                layer = QuantizedActiv('5_Su_Px_QA', layer, self.qa)
                print('q_su : ', layer)

            layer = tf.nn.depth_to_space(layer, 2, data_format="NHWC", name='SR_output')
            print('_fin : ', layer)

        # -- some outputs
        # out_nchw = tf.transpose(layer, [0, 3, 1, 2], name="NCHW_output")
        with tf.variable_scope('psnr'):
            psnr = tf.image.psnr(layer, input_x, max_val=255)
            # print("==========================")
            # print("PSNR: ", psnr)

        mse = tf.losses.mean_squared_error(layer, input_x)
        print("MSE: ", mse)

        # cost related lines...
        add_moving_summary(tf.reduce_mean(mse, name='Mean_Squared_Error'))

        add_moving_summary(psnr)
        add_param_summary(('.*/W', ['histogram']))  # monitor W

        self.cost = tf.add_n([mse], name='cost')

        return self.cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.02, trainable=False)
        # choose which optimizer?
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt

    def loss(self, Y, X):
        dY = tf.image.sobel_edges(Y)
        dX = tf.image.sobel_edges(X)
        M = tf.sqrt(tf.square(dY[:, :, :, :, 0]) + tf.square(dY[:, :, :, :, 1]))
        return tf.losses.absolute_difference(dY, dX) \
               + tf.losses.absolute_difference((1.0 - M) * Y, (1.0 - M) * X, weights=2.0)


def apply(model_path, output_path='.'):
    assert os.path.isfile(config.LOWRES_DIR)
    assert os.path.isdir(output_path)
    lr = cv2.imread(config.LOWRES_DIR).astype(np.float32)
    baseline = cv2.resize(lr, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    LR_SIZE_H, LR_SIZE_W = lr.shape[:2]

    predict_func = OfflinePredictor(PredictConfig(
        model=Model(d=config.FSRCNN_D, s=config.FSRCNN_S, m=config.FSRCNN_M, qw=config.QW, qa=config.QA),
        session_init=SmartInit(model_path),
        input_names=['input_x'],
        output_names=['SR_output']))

    pred = predict_func(lr[None, ...])
    p = np.clip(pred[0][0, ...], 0, 255)

    cv2.imwrite(os.path.join(output_path, "SR_output.png"), p)
    cv2.imwrite(os.path.join(output_path, "baseline.png"), baseline)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load previous model')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU

    logger.set_logger_dir(
        os.path.join('train_log', config.LOG_DIR))

    dataset_train = QueueInput(get_data(config.DATA_ZIP_DIR, 'train'))
    dataset_test = QueueInput(get_data(config.DATA_ZIP_DIR, 'test'))

    if args.apply:
        apply(args.load, config.SROUTPUT_DIR)
    else:
        logger.auto_set_dir()

        train_config = TrainConfig(
            model=Model(d=config.FSRCNN_D, s=config.FSRCNN_S, m=config.FSRCNN_M, qw=config.QW, qa=config.QA),
            data=dataset_train,
            callbacks=[
                ModelSaver(keep_checkpoint_every_n_hours=1),
                InferenceRunner(dataset_test,
                                [ScalarStats('train_error')]),
                ScheduledHyperParamSetter('learning_rate',
                                          [(1, 0.02), (80, 0.002), (160, 0.0002), (300, 0.00002)])
            ],
            max_epoch=config.MAX_EPOCH,
            nr_tower=max(get_num_gpu(), 1),
            session_init=SaverRestore(args.load) if args.load else None
        )
        num_gpu = max(get_num_gpu(), 1)
        launch_train_with_config(train_config, SyncMultiGPUTrainerParameterServer(num_gpu))
