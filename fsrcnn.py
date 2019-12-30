import argparse
import numpy as np
import os
import cv2
import six
import zipfile
import glob

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer

from tensorpack import *
from tensorpack.dataflow.serialize import *
from tensorpack.tfutils import optimizer, gradproc
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger
from tensorpack.utils.gpu import get_num_gpu

import config
from load_data import get_data

from learned_quantization import *

PS = config.CHANNELS * (config.SCALE ** 2)  # for sub-pixel, PS = Phase Shift


def psnr_calc(prediction, ground_truth, maxp=1.0, name='psnr'):
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
    # prediction = tf.abs(prediction)
    # ground_truth = tf.abs(ground_truth)
    def log10(x):
        with tf.name_scope("log10"):
            numerator = tf.log(x)
            denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

    mse = tf.losses.mean_squared_error(prediction, ground_truth)

    if maxp is None:
        psnr = tf.multiply(log10(mse), -10.0)
    else:
        maxp = float(maxp)
        # psnr = tf.multiply(log10(mse + 1e-6), -10.0)
        psnr = tf.multiply(log10(mse), -10.0)
        psnr = tf.add(tf.multiply(20., log10(maxp)), psnr, name=name)
    add_moving_summary(psnr)
    return psnr


class Model(ModelDesc):
    def __init__(self, d, s, m, qw=1, qa=0, height=config.INPUT_IMAGE_SIZE, width=config.INPUT_IMAGE_SIZE):
        super(Model, self).__init__()
        self.height = height
        self.width = width

        self.qw = qw
        self.qa = qa

        self.name = "FSRCNN"

        self.d = d
        self.s = s
        self.m = m

    def inputs(self):
        return [tf.TensorSpec((None, self.width*0.5, self.height*0.5, config.CHANNELS), tf.float32, 'input_lr'),
                tf.TensorSpec((None, self.width, self.height, config.CHANNELS), tf.float32, 'input_hr'),
                tf.TensorSpec((None, self.width, self.height, config.CHANNELS), tf.float32, 'bicubic_hr')
                ]

    def build_graph(self, input_lr, input_hr, bicubic_hr):
        d = self.d
        s = self.s
        m = self.m
        assert tf.test.is_gpu_available()

        with argscope(Conv2D,
                      data_format="NHWC",
                      padding='same', kernel_size=1, stride=1,
                      kernel_initializer=variance_scaling_initializer(mode='FAN_IN'),
                      bias_initializer=tf.constant_initializer(value=0.0),
                      activation=prelu, use_bias=True):
            with argscope(Conv2DQuant,
                          data_format="NHWC",
                          padding='same', kernel_shape=1, stride=1,
                          W_init=variance_scaling_initializer(mode='FAN_IN'),
                          b_init=tf.constant_initializer(value=0.0),
                          nl=prelu, use_bias=True, nbit=self.qw, is_quant=True):
                # kernel_shape = kernel_size
                # W_init = kernel_initializer
                # b_init = bias_initializer
                # nl = activation

                # feature extraction : 5x5 convolutions. (Non-Quantized)
                if config.ORIGINAL_FSRCNN is True:
                    layer = Conv2D('1_Fe_Ex', input_lr, d, kernel_size=5)
                else:
                    layer = Conv2DQuant('1_Fe_Ex', input_lr, d, kernel_shape=5, is_quant=False)
                print('1_Fe_Ex', layer)

                # shrinking : Reduction in feature maps.
                if config.ORIGINAL_FSRCNN is True:
                    layer = Conv2D('2_Shrnk', layer, s)
                else:
                    if self.qa > 0:
                        layer = QuantizedActiv('2_Shrnk_QA', layer, nbit=self.qa)
                    layer = Conv2DQuant('2_Shrnk', layer, s)
                print('2_Shrnk', layer)

                # non-linear mappping : Multiple layers are applied 3x3.
                for i in range(0, m):
                    if config.ORIGINAL_FSRCNN is True:
                        layer = Conv2D('3_Nl_Ma' + str(i), layer, s, kernel_size=3)
                    else:
                        if self.qa > 0:
                            layer = QuantizedActiv('3_Nl_Ma_QA'+str(i), layer, nbit=self.qa)
                        layer = Conv2DQuant('3_Nl_Ma'+str(i), layer, s, kernel_shape=3)
                    print('3_Nl_Ma', layer)

                # expanding : The feature map is now increased by 1x1 convolutions.
                if config.ORIGINAL_FSRCNN is True:
                    layer = Conv2D('4_Expan', layer, d)
                else:
                    if self.qa > 0:
                        layer = QuantizedActiv('4_Expan_QA', layer, self.qa)
                    layer = Conv2DQuant('4_Expan', layer, d)
                print('4_Expan', layer)

                # 1) transposed conv : High resolution image is reconstructed using 9x9 filter.
                # tr_output_shape = calculate_output_shape(layer, 9, 9, 2, 2, 1)
                # layer = tf.nn.conv2d_transpose(name='5_Tr_Cv', input=layer, filters=[9, 9, 1, d],
                #                                output_shape=tr_output_shape,
                #                                strides=2,
                #                                padding='SAME', data_format="NCHW")

                # layer = tf.nn.conv2d_transpose(name="5_Tr_Cv", input=layer, filters=[9.0, 9.0, 1.0, d],
                #                                output_shape=[config.BATCH_SIZE,self.width,self.width,config.CHANNELS],
                #                                strides=[1,2,2,1], padding='SAME', data_format="NHWC")

                # 2) sub-pixel
                if config.ORIGINAL_FSRCNN is True:
                    layer = Conv2D('5_Su_Px', layer, PS, activation=None)
                else:
                    if self.qa > 0:
                        layer = QuantizedActiv('5_Su_Px_QA', layer, self.qa)
                    layer = Conv2DQuant('5_Su_Px', layer, PS, is_quant=False)
                print('5_Su_Px', layer)
                layer = tf.nn.depth_to_space(layer, config.SCALE, data_format="NHWC")

                # bias_initializer = tf.constant_initializer(value=0.0)
                bias = tf.get_variable(shape=[config.CHANNELS], initializer=tf.constant_initializer(value=0.0), name='bias_Su_Px')

                layer = tf.nn.bias_add(layer, bias, data_format="NHWC", name="NHWC_output")
                print('NHWC_output', layer)

        # -- some outputs
        psnr = psnr_calc(layer, input_hr, maxp=config.NORMALIZE)
        psnr_base = psnr_calc(bicubic_hr, input_hr, maxp=config.NORMALIZE, name="psnr_bicubic_baseline")

        mse = tf.losses.mean_squared_error(layer, input_hr)
        print("MSE: ", mse)

        add_param_summary(('.*/W', ['histogram']),  # monitor ../Weight
                          ('.*/b', ['histogram']),  # monitor ../basis
                          ('.*/n', ['histogram']),  # monitor ../new_basis_i
                          ('.*/p', ['histogram']),
                          ('.*/v', ['histogram'])
                          )

        # wd_cost = tf.multiply(config.WEIGHT_DECAY, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        # add_moving_summary(mse, wd_cost)
        # self.cost = tf.add_n([mse, wd_cost], name='add_n_mse_wd_cost')
        self.cost = mse
        return self.cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-3, trainable=False)
        # opt = tf.train.MomentumOptimizer(lr, 0.9)
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        # return opt
        return optimizer.apply_grad_processors(opt, [gradproc.ScaleGradient(('5_Tr_Cv.*', 0.1)), gradproc.SummaryGradient()])


def apply(model_path, output_path='.'):
    psnr_list = []
    psnr_base_list = []
    img_num = 1

    for img in glob.glob(config.LOWRES_DIR+"/*.png"):
        fullimg = cv2.imread(img, 3)
        width = fullimg.shape[0]
        scaled_width = width - (width % config.SCALE)
        height = fullimg.shape[1]
        scaled_height = height - (height % config.SCALE)
        cropped = fullimg[0:scaled_width, 0:scaled_height, :]

        img_ycc = cv2.cvtColor(cropped, cv2.COLOR_BGR2YCrCb)
        if config.CHANNELS == 1:
            # only work on the luminance channel Y
            img_y = img_ycc[:, :, 0]

            input_hr_0 = img_y
            input_hr_norm = input_hr_0.astype(np.float32) / 255.0
            input_hr = np.reshape(input_hr_norm, (1, scaled_width, scaled_height, 1))
        else:
            input_hr_0 = img_ycc
            input_hr_norm = input_hr_0.astype(np.float32) / 255.0
            input_hr = np.reshape(input_hr_norm,(1, scaled_width, scaled_height, 1))

        input_lr_0 = cv2.resize(input_hr_norm, None, fx=1. / config.SCALE, fy=1. / config.SCALE, interpolation=cv2.INTER_CUBIC)
        input_lr = np.reshape(input_lr_0, (1, input_lr_0.shape[0], input_lr_0.shape[1], 1))

        bicubic_hr_0 = cv2.resize(input_lr_0, None, fx=1.*config.SCALE, fy=1.*config.SCALE, interpolation=cv2.INTER_CUBIC)
        bicubic_hr = np.reshape(bicubic_hr_0, (1, scaled_width, scaled_height, 1))

        pred_config = PredictConfig(model=Model(d=config.FSRCNN_D,
                                                s=config.FSRCNN_S,
                                                m=config.FSRCNN_M,
                                                qw=config.QW,
                                                qa=config.QA,
                                                height=scaled_height,
                                                width=scaled_width
                                                ),
                                    session_init=SmartInit(model_path),
                                    input_names=['input_lr', 'input_hr', 'bicubic_hr'],
                                    output_names=['NHWC_output', 'psnr', 'input_lr', 'psnr_bicubic_baseline'])

        predictor = OfflinePredictor(pred_config)

        NHWC_output, psnr, input_lr_out, psnr_base = predictor(input_lr, input_hr, bicubic_hr)

        NHWC_output = (NHWC_output * 255.0).clip(min=0, max=255)
        NHWC_output = (NHWC_output).astype(np.uint8)
        NHWC_output = np.reshape(NHWC_output, (scaled_width, scaled_height, 1))

        print("PSNR of fsrcnn  upscaled image: {}".format(psnr))
        print("PSNR of bicubic upscaled image: {}".format(psnr_base))

        img_y_ex = np.expand_dims(img_y, axis=3)
        bicubic_hr_out = bicubic_hr_0.astype(np.float32) * 255.0

        if config.ORIGINAL_FSRCNN is True:
            cv2.imwrite(output_path+str(img_num)+'_orgn'+'_psnr('+ str(psnr)+")_fsrcnnOutput.png",NHWC_output)
            cv2.imwrite(output_path+str(img_num)+'_orgn'+'_psnrbase(' + str(psnr_base) + ")_bicubicOutput.png", bicubic_hr_out)
            cv2.imwrite(output_path+str(img_num)+'_orgn'+'_input.png', img_y_ex)
        else:
            cv2.imwrite(output_path+str(img_num)+'_qa'+str(config.QA)+'_qw'+str(config.QW)+'_psnr('+str(psnr)+")_fsrcnnOutput.png", NHWC_output)
            cv2.imwrite(output_path+str(img_num)+'_qa'+str(config.QA)+'_qw'+str(config.QW)+'_psnrbase('+str(psnr_base)+")_bicubicOutput.png", bicubic_hr_out)
            cv2.imwrite(output_path+str(img_num)+'_qa'+str(config.QA)+'_qw'+str(config.QW)+'_input.png', img_y_ex)
        psnr_list.append(psnr)
        psnr_base_list.append(psnr_base)
        img_num = img_num+1

    psnr_avg = sum(psnr_list) / len(psnr_list)
    psnr_base_avg = sum(psnr_base_list) / len(psnr_base_list)
    print("Average PSNR of fsrcnn  upscaled image: {}".format(psnr_avg))
    print("Average PSNR of bicubic upscaled image: {}".format(psnr_base_avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load previous model')
    parser.add_argument('--apply', action='store_true')

    # parser.add_argument("--data_zip_dir", default='/database/saehyun/parasite/General-100_comp/General-100.zip')
    # parser.add_argument("--log_dir", default='/home/saehyun/parasite/pycharm_project/tensorpack_study/log_fsrcnn_orgn/')
    # parser.add_argument("--gpu", default="0, 1")
    # parser.add_argument("--dataflow_proc", default=2)
    # parser.add_argument("--lowres_dir", default='/home/saehyun/parasite/pycharm_project/tensorpack_study/test.zip')
    # parser.add_argument("--sroutput_dir", default='/home/saehyun/parasite/pycharm_project/tensorpack_study/')
    #
    # parser.add_argument("--batch_size", default=4)
    # parser.add_argument("--weight_decay", default=5e-4)
    # parser.add_argument("--max_epoch", default=1600)
    # parser.add_argument("--fsrcnn_d", default=56)
    # parser.add_argument("--fsrcnn_s", default=12)
    # parser.add_argument("--fsrcnn_m", default=4)
    #
    # parser.add_argument("--input_image_size", default=100)
    # parser.add_argument("--normalize", default=1.0)
    # parser.add_argument("--use_ycbcr", default=True)
    # parser.add_argument("--channels", default=1)
    #
    # parser.add_argument("--scale", default=2)
    # parser.add_argument("--qa", default=0)
    # parser.add_argument("--qw", default=1)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU

    logger.set_logger_dir(
        os.path.join('logger_log', config.LOG_DIR))

    dataset_train = QueueInput(get_data(config.TRAIN_DATA_ZIP_DIR, 'train'))
    dataset_test = QueueInput(get_data(config.VAL_DATA_ZIP_DIR, 'test'))

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
                                [ScalarStats('mean_squared_error/value')],
                                ),
                # ScheduledHyperParamSetter('learning_rate',
                #                           [(1, 1e-4), (100, 1e-5), (160, 1e-6), (300, 1e-7)])
                ScheduledHyperParamSetter('learning_rate',
                                          [(1, 1e-3)])
            ],
            max_epoch=config.MAX_EPOCH,
            nr_tower=max(get_num_gpu(), 1),
            session_init=SaverRestore(args.load) if args.load else None
        )
        num_gpu = max(get_num_gpu(), 1)
        launch_train_with_config(train_config, SyncMultiGPUTrainerParameterServer(num_gpu))

