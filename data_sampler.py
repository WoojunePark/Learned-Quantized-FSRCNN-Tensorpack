import argparse
import numpy as np
import os
import zipfile
import cv2

from tensorpack import MapDataComponent, RNGDataFlow
from tensorpack.dataflow.serialize import LMDBSerializer
from tensorpack.dataflow.imgaug.base import PhotometricAugmentor

import config


class ImageDataFromZIPFile(RNGDataFlow):
    """ Produce images read from a list of zip files. """
    def __init__(self, zip_file, shuffle=False):
        """
        Args:
            zip_file (list): list of zip file paths.
        """
        assert os.path.isfile(zip_file)
        self._file = zip_file
        self.shuffle = shuffle
        self.open()

    def open(self):
        self.archivefiles = []
        archive = zipfile.ZipFile(self._file)
        imagesInArchive = archive.namelist()
        img_extension = imagesInArchive[0][-4:]
        for img_name in imagesInArchive:
            if img_name.endswith(img_extension):
                self.archivefiles.append((archive, img_name))

    def reset_state(self):
        super(ImageDataFromZIPFile, self).reset_state()
        # Seems necessary to reopen the zip file in forked processes.
        self.open()

    def size(self):
        return len(self.archivefiles)

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.archivefiles)
        for archive in self.archivefiles:
            im_data = archive[0].read(archive[1])
            im_data = np.asarray(bytearray(im_data), dtype='uint8')
            yield [im_data]


class ImageEncode(MapDataComponent):
    def __init__(self, ds, mode='.jpg', dtype=np.uint8, index=0):
        def func(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return np.asarray(bytearray(cv2.imencode(mode, img)[1].tostring()), dtype=dtype)
        super(ImageEncode, self).__init__(ds, func, index=index)


class ImageDecodeBGR(MapDataComponent):
    def __init__(self, ds, index=0):
        def func(im_data):
            img = cv2.imdecode(im_data, cv2.IMREAD_COLOR)
            return img
        super(ImageDecodeBGR, self).__init__(ds, func, index=index)


class ImageDecodeYCrCb(MapDataComponent):
    def __init__(self, ds, index=0):
        def func(im_data):
            # set lr and hr sizes
            size_lr = 10
            if config.SCALE == 3:
                size_lr = 7
            elif config.SCALE == 4:
                size_lr = 6
            size_hr = size_lr * config.SCALE

            img = cv2.imdecode(im_data, cv2.IMREAD_COLOR)
            img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

            if config.CHANNELS == 1:
                lr_y = img_ycc[:, :, 0]
                lr_y_ex = np.expand_dims(lr_y, axis=3)
                return lr_y_ex
            else:
                return img_ycc

        super(ImageDecodeYCrCb, self).__init__(ds, func, index=index)


class RejectTooSmallImages(MapDataComponent):
    def __init__(self, ds, thresh=100, index=0):
        def func(img):
            h, w = img.shape
            if (h < thresh) or (w < thresh):
                return None
            else:
                return img
        super(RejectTooSmallImages, self).__init__(ds, func, index=index)


class CenterSquareResize(MapDataComponent):
    def __init__(self, ds, index=0):
        def func(img):
            try:
                h, w, _ = img.shape
                if h > w:
                    off = (h - w) // 2
                    if off > 0:
                        img = img[off:-off, :, :]
                if w > h:
                    off = (w - h) // 2
                    if off > 0:
                        img = img[:, off:-off, :]

                img = cv2.resize(img, (100, 100))
                return img
            except Exception:
                return None
        super(CenterSquareResize, self).__init__(ds, func, index=index)


class MinMaxNormalize(PhotometricAugmentor):
    """
    Linearly scales the image to the range [min, max].
    This augmentor always returns float32 images.
    """
    def __init__(self, min=0, max=255, all_channel=True):
        """
        Args:
            max (float): The new maximum value
            min (float): The new minimum value
            all_channel (bool): if True, normalize all channels together. else separately.
        """
        self.max = max
        self.min = min
        self.all_channel = all_channel
        self._init(locals())

    def _augment(self, img, _):
        img = img.astype('float32')
        if self.all_channel:
            minimum = np.amin(img)
            maximum = np.amax(img)
        else:
            minimum = np.amin(img, initial=self.max, keepdims=True)
            maximum = np.amax(img, initial=self.min, keepdims=True)

        img = (self.max - self.min) * (img - minimum) / (maximum - minimum) + self.min

        return img


# Testcode for encode/decode.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--create', action='store_true', help='create lmdb')
    parser.add_argument('--debug', action='store_true', help='debug images')
    parser.add_argument('--input', type=str, help='path to coco zip', required=True)
    parser.add_argument('--lmdb', type=str, help='path to output lmdb', required=True)
    args = parser.parse_args()

    ds = ImageDataFromZIPFile(args.input)
    ds = ImageDecodeYCrCb(ds, index=0)
    # ds = RejectTooSmallImages(ds, index=0)
    ds = CenterSquareResize(ds, index=0)
    if args.create:
        ds = ImageEncode(ds, index=0)
        LMDBSerializer.save(ds, args.lmdb)
    if args.debug:
        ds.reset_state()
        for i in ds:
            cv2.imshow('example', i[0])
            cv2.waitKey(0)
