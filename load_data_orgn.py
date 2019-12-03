import argparse
import numpy as np
import os
import sys
import cv2
import six
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorpack import *
from tensorpack.dataflow import *
from tensorpack.dataflow.serialize import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger
from tensorpack.utils.gpu import get_num_gpu

import imageio
import time

from data_sampler_orgn import CenterSquareResize, ImageDataFromZIPFile, \
    ImageDecodeYCrCb, ImageDecodeBGR, RejectTooSmallImages, MinMaxNormalize, ThreeInputs

import config_orgn as config
import learned_quantization


def get_data(file_name, train_or_test):
    isTrain = train_or_test == 'train'

    if file_name.endswith('.lmdb'):
        ds = LMDBSerializer.load(file_name, shuffle=True)
        if config.USE_YCBCR is True:
            ds = ImageDecodeYCrCb(ds, index=0)
        else:
            ds = ImageDecodeBGR(ds, index=0)
    elif file_name.endswith('.zip'):
        ds = ImageDataFromZIPFile(file_name, shuffle=True)
        if config.USE_YCBCR is True:
            ds = ImageDecodeYCrCb(ds, index=0)
        else:
            ds = ImageDecodeBGR(ds, index=0)
        # ds = RejectTooSmallImages(ds, thresh=100, index=0)
        # ds = CenterSquareResize(ds, index=0)
    else:
        raise ValueError("Unknown file format " + file_name)

    if isTrain:
        augmentors = [
            # imgaug.ToFloat32,
            # MinMaxNormalize(0, 255, all_channel=False),
            imgaug.RandomCrop(100),
            # imgaug.RandomApplyAug(imgaug.RandomChooseAug([
            #     imgaug.SaltPepperNoise(white_prob=0.01, black_prob=0.01),
            #     imgaug.RandomOrderAug([
            #         imgaug.BrightnessScale((0.98, 1.02), clip=True),
            #         # imgaug.Contrast((0.98, 1.02), rgb=None, clip=True),
            #         # imgaug.Saturation(0.4, rgb=False),  # only for RGB or BGR images!
            #         ]),
            #     ]), 0.7),
            # imgaug.SaltPepperNoise(white_prob=0.01, black_prob=0.01),
            imgaug.RandomApplyAug(
                imgaug.RandomOrderAug([
                    imgaug.Flip(horiz=True),
                    imgaug.Flip(vert=True),
                    # imgaug.BrightnessScale((0.98, 1.02), clip=True),
                    # imgaug.Contrast((0.98, 1.02), rgb=None, clip=True),
                    # imgaug.Saturation(0.4, rgb=False),  # only for RGB or BGR images!
                    ]),
                0.7),

            # imgaug.MinMaxNormalize(0.0001, config.NORMALIZE, all_channel=True),
            MinMaxNormalize(0, config.NORMALIZE, all_channel=False),
        ]
    else:
        augmentors = [
            imgaug.RandomCrop(100),
            imgaug.MinMaxNormalize(0, config.NORMALIZE, all_channel=False),
        ]

    ds = AugmentImageComponent(ds, augmentors, index=0, copy=True)

    #if isTrain:
    #   ds = PrefetchData(ds, 2, 2)
    ds = MapData(ds, lambda x: [np.expand_dims(cv2.resize(x[0], (50, 50), interpolation=cv2.INTER_CUBIC), axis=3),
                                x[0],
                                np.expand_dims(cv2.resize(cv2.resize(x[0], (50, 50), interpolation=cv2.INTER_CUBIC),
                                           (100, 100), interpolation=cv2.INTER_CUBIC),axis=3),
                                ])

    ds = MultiProcessRunnerZMQ(ds, config.DATAFLOW_PROC)
    ds = BatchData(ds, config.BATCH_SIZE, remainder=not isTrain)
    return ds


def _get_data_readonly(file_name, train_or_test):
    print("_get_data_readonly isTrain: ", train_or_test)
    ds = get_data(file_name, train_or_test)
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help="'speed' for dataflow loading speed test, 'shape' for shape test",
                        type=str, default='speed')
    args = parser.parse_args()

    isSpeed = args.test == 'speed'
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
    if isSpeed is True:
        # for dataflow loading speed test
        # from tensorpack.dataflow.common import TestDataSpeed
        ds = _get_data_readonly(config.DATA_ZIP_DIR, 'train')
        # size doesn't matter unless < all_data/batch.
        TestDataSpeed(ds, size=10).start()
        print("speed test done!")
        sys.exit(0)

    else:
        im = _get_data_readonly(config.DATA_ZIP_DIR, 'train')
        im.reset_state()
        for i in im:
            print(np.shape(np.array(i)))
            time_name = time.ctime()
            time_name += '.jpg'
            imageio.imwrite(time_name, i[0,0, :, :, 0])
        print("shape test done!")
        sys.exit(0)
#
        """
        # >>>
        # (1, 5, 100, 100, 3)
        # (1, 5, 100, 100, 3)
        # (1, 5, 100, 100, 3)
        # ...
        # (?, batch size, h, w, c) x (# of data/batch size)
        """
