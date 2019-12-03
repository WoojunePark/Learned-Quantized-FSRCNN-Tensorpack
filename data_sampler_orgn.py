import argparse
import numpy as np
import os
import zipfile
import cv2

from tensorpack import MapDataComponent, RNGDataFlow
from tensorpack.dataflow.serialize import LMDBSerializer

from tensorpack.dataflow.imgaug.base import PhotometricAugmentor
import imageio
import time

import config_orgn as config


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
                # isn't imagesInArchive already a list? why go through troubles?...

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
            # im_data_hr = cv2.imdecode(im_data, cv2.IMREAD_COLOR)
            #
            # im_data_lr = cv2.resize(im_data, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
            # im_data_hrb = cv2.resize(im_data_lr, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
            # yield [im_data_lr, im_data_hr, im_data_hrb]
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
            # print("type of img is : ", type(img))
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

            # read
            img = cv2.imdecode(im_data, cv2.IMREAD_COLOR)

            # convert to YCrCb (cv2 reads images in BGR!), and normalize
            img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            # img_ycc = cv2.cvtColor(im_data, cv2.COLOR_BGR2YCrCb)

            # # resized, original, bicubic&bicubic
            # lr_ycc = cv2.resize(img_ycc, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
            # hr_bicubic_ycc = cv2.resize(lr_ycc, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)

            if config.CHANNELS == 1:
                # input_bicubic_y, input_bicubic_cr, input_bicubic_cb = cv2.split(input_bicubic_ycc)

                # only work on the luminance channel Y
                lr_y = img_ycc[:, :, 0]
                # hr_y = img_ycc[:, :, 0]
                # hr_bicubic_y = hr_bicubic_ycc[:, :, 0]

                # (1, 4, 100, 100, 1)
                # im_y[:,0:0,:,:,:]
                lr_y_ex = np.expand_dims(lr_y, axis=3)
                # hr_y_ex = np.expand_dims(hr_y, axis=4)
                # hr_bicubic_y_ex = np.expand_dims(hr_bicubic_y, axis=4)
                return lr_y_ex
            else:
                return img_ycc

        super(ImageDecodeYCrCb, self).__init__(ds, func, index=index)


class ThreeInputs(RNGDataFlow):
    def __init__(self, ds, index=0):
        # resized, original, bicubic&bicubic
        lr = cv2.resize(ds, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
        hr_bicubic = cv2.resize(lr, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)

    def __len__(self):
        return self.img.shape[0]

    def __iter__(self):
        yield [self.lr, self.ds, self.hr_bicubic]

        # super(ThreeInputs, self).__init__(ds, func, index=index)


class RejectTooSmallImages(MapDataComponent):
    def __init__(self, ds, thresh=100, index=0):
        def func(img):
            # (50, 50) and (100, 100) at the same time version
            # h0, w0, _ = img[0].shape
            # h1, w1, _ = img[1].shape
            # if (h1 < thresh) or (w1 < thresh):
            #     return None
            # else:
            #     return img

            # 1 img version
            h, w = img.shape
            if (h < thresh) or (w < thresh):
                return None
            else:
                return img
        super(RejectTooSmallImages, self).__init__(ds, func, index=index)


class CenterSquareResize(MapDataComponent):
    def __init__(self, ds, index=0):
        """See section 5.3
        """
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
            # minimum = np.min(img)
            # maximum = np.max(img)
            minimum = np.amin(img)
            maximum = np.amax(img)

        else:
            # minimum = np.min(img, axis=(0, 1), keepdims=True)
            # maximum = np.max(img, axis=(0, 1), keepdims=True)
            minimum = np.amin(img, initial=self.max, keepdims=True)
            maximum = np.amax(img, initial=self.min, keepdims=True)

        # time_name = time.ctime()
        # time_name += '.jpg'
        # imageio.imwrite(time_name, img[:, :, 0])

        # if (maximum - minimum) < 1e-10:
        #     print("adsdasdasdsadasdasd")
        #     imageio.imwrite('poped_from_MinMaxNormalize.jpg', img[:, :, 0])

        img = (self.max - self.min) * (img - minimum) / (maximum - minimum) + self.min

        # imageio.imwrite('poped_from_MinMaxNormalize.jpg', img[:, :, 0])

        return img


def make_dataset(paths):
    """
    Python generator-style dataset. Creates low-res and corresponding high-res patches.
    """
    # set lr and hr sizes
    size_lr = 10
    if config.SCALE == 3:
        size_lr = 7
    elif config.SCALE == 4:
        size_lr = 6
    size_hr = size_lr * config.SCALE

    for p in paths:
        # read
        im = cv2.imread(p.decode(), 3).astype(np.float32)

        # convert to YCrCb (cv2 reads images in BGR!), and normalize
        im_ycc = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb) / 255.0

        # -- Creating LR and HR images
        # make current image divisible by scale (because current image is the HR image)
        im_ycc_hr = im_ycc[0:(im_ycc.shape[0] - (im_ycc.shape[0] % config.SCALE)),
                    0:(im_ycc.shape[1] - (im_ycc.shape[1] % config.SCALE)), :]
        im_ycc_lr = cv2.resize(im_ycc_hr, (int(im_ycc_hr.shape[1] / config.SCALE),
                                           int(im_ycc_hr.shape[0] / config.SCALE)),
                               interpolation=cv2.INTER_CUBIC)

        # only work on the luminance channel Y
        lr = im_ycc_lr[:, :, 0]
        hr = im_ycc_hr[:, :, 0]

        numx = int(lr.shape[0] / size_lr)
        numy = int(lr.shape[1] / size_lr)

        for i in range(0, numx):
            startx = i * size_lr
            endx = (i * size_lr) + size_lr

            startx_hr = i * size_hr
            endx_hr = (i * size_hr) + size_hr

            for j in range(0, numy):
                starty = j * size_lr
                endy = (j * size_lr) + size_lr
                starty_hr = j * size_hr
                endy_hr = (j * size_hr) + size_hr

                crop_lr = lr[startx:endx, starty:endy]
                crop_hr = hr[startx_hr:endx_hr, starty_hr:endy_hr]

                x = crop_lr.reshape((size_lr, size_lr, 1))
                y = crop_hr.reshape((size_hr, size_hr, 1))
                yield x, y


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
