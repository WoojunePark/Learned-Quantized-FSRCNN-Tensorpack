import cv2
import numpy as np
import tensorflow as tf
import os
import glob
import h5py


# Get the Image
def img_read(path):
    img = cv2.imread(path)
    return img


def img_save(image, path, config):
    # Check the result dir, if not, create one
    if not os.path.isdir(os.path.join(os.getcwd(),config.result_dir)):
        os.makedirs(os.path.join(os.getcwd(),config.result_dir))
    print(os.path.join(os.getcwd(), path))
    # NOTE: because normial, we need mutlify 255 back
    cv2.imwrite(os.path.join(os.getcwd(),path),image * 255.)


def img_check(image):
    cv2.imshow("test",image)
    cv2.waitKey(0)


def modcrop(img, scale =3):
    """
        To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
    """
    # Check the image is grayscale
    if len(img.shape) ==3:
        h, w, _ = img.shape
        h = (h / scale) * scale
        w = (w / scale) * scale
        img = img[0:h, 0:w, :]
    else:
        h, w = img.shape
        h = (h / scale) * scale
        w = (w / scale) * scale
        img = img[0:h, 0:w]
    return img


def checkpoint_dir(config):
    if config.is_train:
        return os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
    else:
        return os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")


def preprocess(path, scale=3):
    """
        Args:
            path: the image directory path
            scale: the image need to scale
    """
    img = imread(path)

    label_ = modcrop(img, scale)

    input_ = cv2.resize(label_, None, fx=1.0 / scale, fy=1.0 / scale,
                        interpolation=cv2.INTER_AREA)  # Resize by scaling factor

    return input_, label_