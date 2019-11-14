import scipy.misc
import random
import numpy as np
import os
import config

train_set = []
test_set = []
batch_index = 0

"""
Load set of images in a directory.
This will automatically allocate a 
random 20% of the images as a test set

data_dir: path to directory containing images

reference : https://github.com/jmiller656/EDSR-Tensorflow/blob/master/data.py
"""


def load_dataset(data_dir, img_size):
    """
    img_files = os.listdir(data_dir)
    test_size = int(len(img_files)*0.2)
    test_indices = random.sample(range(len(img_files)),test_size)
    for i in range(len(img_files)):
        #img = scipy.misc.imread(data_dir+img_files[i])
        if i in test_indices:
            test_set.append(data_dir+"/"+img_files[i])
        else:
            train_set.append(data_dir+"/"+img_files[i])
    return
    """
    global train_set
    global test_set
    img_loaded = []
    img_files = os.listdir(data_dir)
    for img in img_files:
        try:
            tmp = scipy.misc.imread(data_dir)
            x, y, z = tmp.shape
            coord_x = x / img_size
            coord_y = x / img_size
            coords = [(q, r) for q in range(coord_x) for r in range(coord_y)]
            for coord in coords:
                img.append((data_dir + "/" + img, coord))
        except:
            print("load_dataset/for img in img_files/except!!")

    test_size = min(10, int(len(img_loaded)*0.2))
    random.shuffle(img_loaded)
    test_set = img_loaded[:test_size]  # slice : 0 -> (test_size)-1
    train_set = img_loaded[:test_size][:200]  # slice : 0 -> (test_size)-1 and # slice : 0 -> 199th
    return


"""
Get test set from the loaded dataset
size (optional): if this argument is chosen,
each element of the test set will be cropped
to the first (size x size) pixels in the image.

returns the test set of your data
"""


def get_image(img_in_tuple, size):
    img = scipy.misc.imread(img_in_tuple[0])
    x, y = img_in_tuple[1]
    img = img[x*size:(x+1)*size, y*size:(y+1)*size]
    return img


def get_test_set(original_size, shrunk_size):
    """
    for i in range(len(test_set)):
    img = scipy.misc.imread(test_set[i])
        if img.shape:
            img = crop_center(img,original_size,original_size)
            x_img = scipy.misc.imresize(img,(shrunk_size,shrunk_size))
            y_imgs.append(img)
            x_imgs.append(x_img)
    """
    img_loaded = test_set
    get_image(img_loaded[0],original_size)
    x = [scipy.misc.imresize(get_image(q, original_size), (shrunk_size, shrunk_size)) for q in img_loaded]
    # scipy.misc.imread(q[0])[q[1][0]*original_size:(q[1][0]+1)*original_size,q[1][1]*original_size:(q[1][1]+1)*original_size].resize(shrunk_size,shrunk_size) for q in imgs]
    y = [get_image(q, original_size) for q in img_loaded]
    # scipy.misc.imread(q[0])[q[1][0]*original_size:(q[1][0]+1)*original_size,q[1][1]*original_size:(q[1][1]+1)*original_size] for q in imgs]
    return x, y


"""
Get a batch of images from the training
set of images.

batch_size: size of the batch
original_size: size for target images
shrunk_size: size for shrunk images

returns x,y where:
-x is the input set of shape [-1,shrunk_size,shrunk_size,channels]
-y is the target set of shape [-1,original_size,original_size,channels]
"""






