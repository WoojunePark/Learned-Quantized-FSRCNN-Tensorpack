import argparse
import numpy as np
import os
import sys
import cv2
import six
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import *
from tensorpack.dataflow.serialize import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger
from tensorpack.utils.gpu import get_num_gpu

from data_sampler import CenterSquareResize, ImageDataFromZIPFile, ImageDecode, RejectTooSmallImages

import config
import learned_quantization


def get_data(file_name, train_or_test):
    isTrain = train_or_test == 'train'
    if file_name.endswith('.lmdb'):
        ds = LMDBSerializer.load(file_name, shuffle=True)
        ds = ImageDecode(ds, index=0)
    elif file_name.endswith('.zip'):
        ds = ImageDataFromZIPFile(file_name, shuffle=True)
        ds = ImageDecode(ds, index=0)
        ds = RejectTooSmallImages(ds, thresh=100, index=0)
        # ds = CenterSquareResize(ds, index=0)
    else:
        raise ValueError("Unknown file format " + file_name)

    if isTrain:
        augmentors = [
            imgaug.RandomCrop(100),
            imgaug.Flip(horiz=True),
            imgaug.RandomApplyAug(imgaug.RandomChooseAug([
                imgaug.SaltPepperNoise(white_prob=0.01, black_prob=0.01),
                imgaug.RandomOrderAug([
                    imgaug.BrightnessScale((0.8, 1.2), clip=False),
                    imgaug.Contrast((0.8, 1.2), clip=False),
                    # imgaug.Saturation(0.4, rgb=True),
                    ]),
                ]), 0.7)
        ]
    else:
        augmentors = [
            imgaug.RandomCrop(100),
        ]
    ds = AugmentImageComponent(ds, augmentors, index=0, copy=True)

    # ds = MapData(ds, lambda x: [cv2.resize(x[0], (32, 32), interpolation=cv2.INTER_CUBIC), x[0]])
    ds = MultiProcessRunnerZMQ(ds, config.DATAFLOW_PROC)
    ds = BatchData(ds, config.BATCH_SIZE, remainder=not isTrain)
    # ds = PrefetchData(ds, 3, 2)
    return ds


def _get_data_readonly(file_name, train_or_test):
    isTrain = train_or_test == 'train'
    ds = get_data(file_name, isTrain)
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
        print("shape test done!")

        """
        # >>>
        # (1, 5, 100, 100, 3)
        # (1, 5, 100, 100, 3)
        # (1, 5, 100, 100, 3)
        # ...
        # (?, batch size, h, w, c) x (# of data/batch size)
        """
