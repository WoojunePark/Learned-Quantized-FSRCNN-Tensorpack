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
CHANNELS = 1
PS = CHANNELS * config.SCALE * config.SCALE   # for sub-pixel, PS = Phase Shift


def psnr_calc(prediction, ground_truth, maxp=None, name='psnr'):
    """`Peek Signal to Noise Ratio <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`_.
    .. math::
        PSNR = 20 \cdot \log_{10}(MAX_p) - 10 \cdot \log_{10}(MSE)
    Args:
        prediction: a :class:`tf.Tensor` representing the prediction signal.
        ground_truth: another :class:`tf.Tensor` with the same shape.
        maxp: maximum possible pixel value of the image (255 in in 8bit images)
    Returns:
        A scalar tensor representing the PSNR. (tf.psnr returns (psnr_value, 1) tensor.)
    """
    prediction = tf.abs(prediction)
    ground_truth = tf.abs(ground_truth)

    def log10(x):
        with tf.name_scope("log10"):
            numerator = tf.log(x)
            denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

    mse = tf.reduce_mean(tf.square(prediction - ground_truth))
    if maxp is None:
        psnr = tf.multiply(log10(mse), -10., name=name)
    else:
        maxp = float(maxp)
        psnr = tf.multiply(log10(mse + 1e-6), -10.)
        psnr = tf.add(tf.multiply(20., log10(maxp)), psnr, name=name)
    add_moving_summary(psnr)
    return psnr


class Model(ModelDesc):
    def __init__(self, d, s, m, qw=1, qa=0, height=SHAPE_LR, width=SHAPE_LR):
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
        return [tf.TensorSpec((None, self.height*1,self.width*1, CHANNELS), tf.float32, 'input_x'),
                # tf.TensorSpec((None, self.height*4,self.width*4, CHANNELS), tf.float32, 'input_hr')
                ]

    def build_graph(self, input_x):
        # input_x = input_x / 255.0
        # image_size = config.INPUT_IMAGE_SIZE

        d = self.d
        s = self.s
        m = self.m
        input_bicubic = tf.image.resize(
            input_x, [50, 50], method=tf.image.ResizeMethod.BICUBIC,
            name='bicubic_baseline')
        # input_x = original 100X100 image
        # input_bicubic = resized 50x50 image

        assert tf.test.is_gpu_available()
        input_bicubic = tf.transpose(input_bicubic, [0, 3, 1, 2])  # NHWC to NCHW

        # with argscope([Conv2DQuant, MaxPooling, BatchNorm], ), \
        with argscope(Conv2DQuant,
                      data_format="NCHW",
                      padding='same', kernel_shape=1, stride=1,
                      W_init=variance_scaling_initializer(mode='FAN_IN'),
                      # b_init=tf.constant_initializer(value=0.0),
                      nl=PReLU_4Q, use_bias=False, is_quant=True if self.qw > 0 else False, nbit=self.qw):

            # -- Model architecture --
            # feature extraction : 5x5 convolutions. (Non-Quantized)
            # layer = Conv2D('1_Fe_Ex', input_bicubic, [5, 5, 3, d], stride=1, padding='same', use_bias=False)
            layer = Conv2DQuant('1_Fe_Ex', input_bicubic, d, kernel_shape=5, is_quant=False)
            print('1_Fe_Ex', layer)

            # shrinking : Reduction in feature maps.
            if self.qa > 0:
                layer = QuantizedActiv('2_Shrnk_QA', layer, nbit=self.qa)
            layer = Conv2DQuant('2_Shrnk', layer, s)
            print('2_Shrnk', layer)

            # non-linear mappping : Multiple layers are applied 3x3.
            for i in range(0, m):
                if self.qa > 0:
                    layer = QuantizedActiv('3_Nl_Ma_QA'+str(i), layer, nbit=self.qa)
                layer = Conv2DQuant('3_Nl_Ma'+str(i), layer, s, kernel_shape=3)
                print('3_Nl_Ma', layer)

            # expanding : The feature map is now increased by 1x1 convolutions.
            if self.qa > 0:
                layer = QuantizedActiv('4_Expan_QA', layer, self.qa)
            layer = Conv2DQuant('4_Expan', layer, d)
            print('4_Expan', layer)

            # 1) transposed conv : High resolution image is reconstructed using 9x9 filter.
            # layer = tf.nn.conv2d_transpose('5_Tr_Cv', layer, filters=[9, 9, 3, d],
            #                                output_shape=[config.BATCH_SIZE, 100, 100, 3],
            #                                strides=2,
            #                                padding='SAME', data_format="NHWC")

            # 2) sub-pixel
            if self.qa > 0:
                layer = QuantizedActiv('5_Su_Px_QA', layer, self.qa)
            layer = Conv2DQuant('5_Su_Px', layer, PS, is_quant=False)
            print('5_Su_Px', layer)

            layer = tf.nn.depth_to_space(layer, config.SCALE, data_format="NCHW", name='SR_output')
            print('SR_output', layer)

        # -- some outputs
        out_nhwc = tf.transpose(layer, [0, 2, 3, 1], name="NHWC_output")  # From NCHW to NHWC
        psnr = psnr_calc(out_nhwc, input_x, maxp=255)
        print("PSNR: ", psnr)
        add_moving_summary(tf.reduce_mean(psnr, name='psnr_sum'))

        mse = tf.losses.mean_squared_error(out_nhwc, input_x)
        print("MSE: ", mse)

        add_moving_summary(tf.reduce_mean(mse, name='mean_squared_error_sum'))

        add_param_summary(('.*/W', ['histogram']))  # monitor W
        self.cost = tf.add_n([mse], name='mean_squared_error_inf')

        return self.cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.02, trainable=False)
        # choose which optimizer?
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt


def apply(model_path, output_path='.'):
    assert os.path.isfile(config.LOWRES_DIR)
    assert os.path.isdir(output_path)

    input_x = get_data(config.LOWRES_DIR, 'test')
    input_bicubic = tf.image.resize(
        input_x, [50, 50], method=tf.image.ResizeMethod.BICUBIC,
        name='input_x')

    predict_func = OfflinePredictor(PredictConfig(
        model=Model(d=config.FSRCNN_D, s=config.FSRCNN_S, m=config.FSRCNN_M, qw=config.QW, qa=config.QA),
        session_init=SmartInit(model_path),
        input_names=['input_x'],
        output_names=['NHWC_output']))

    pred = predict_func(input_bicubic[None, ...])
    p = np.clip(pred[0][0, ...], 0, 255)

    cv2.imwrite(os.path.join(output_path, "SR_output.png"), pred)
    cv2.imwrite(os.path.join(output_path, "input_bicubic.png"), input_x)


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
                                [ScalarStats('mean_squared_error_inf')]),
                ScheduledHyperParamSetter('learning_rate',
                                          [(1, 0.02), (80, 0.002), (160, 0.0002), (300, 0.00002)])
            ],
            max_epoch=config.MAX_EPOCH,
            nr_tower=max(get_num_gpu(), 1),
            session_init=SaverRestore(args.load) if args.load else None
        )
        num_gpu = max(get_num_gpu(), 1)
        launch_train_with_config(train_config, SyncMultiGPUTrainerParameterServer(num_gpu))
