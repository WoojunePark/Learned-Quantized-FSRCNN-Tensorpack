import cv2           # for AugmentImageComponent
import multiprocessing
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorpack import *
from tensorpack.dataflow import *
from tensorpack.tfutils import summary

# for data loading
# building a model
# training & test
# transfer learning
# eval


def get_dataflow(batch_size, is_train='train'):
    """
    augment your data here
    :param batch_size:
    :param is_train:
    :return:
    """
    df = dataset.Mnist(is_train, shuffle=True)
    istrain = is_train == 'train'

    # ----- Image Augmentation Options -------- #
    if istrain:
        augs = [imgaug.CenterCrop((256, 256)),
                imgaug.Resize((225, 225)),
                imgaug.Grayscale(keepdims=True),
                imgaug.Flip(horiz=True, vert=False, prob=0.5)
                ]
    else:
        # for testing
        augs = [imgaug.CenterCrop((256, 256)),
                imgaug.Resize((225, 225))
                ]

    df = AugmentImageComponent(df, augs)
    # group data into batches of size 128
    df = BatchData(df, batch_size)
    # start 3 processes to run the dataflow in parallel
    # df = PrefetchDataZMQ(df, 10, multiprocessing.cpu_count())
    return df


df = get_dataflow(4, 'train')