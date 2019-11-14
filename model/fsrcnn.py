import argparse

from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_num_gpu

import config
from learned_quantization import *

NUM_UNIT = None


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, config.BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds


class Model(ModelDesc):
    def __init__(self, qw=1, qa=1):
        super(Model, self).__init__()
        self.qw = qw
        self.qa = qa

    def inputs(self):
        return [tf.TensorSpec([None, config.INPUT_IMAGE_SIZE, config.INPUT_IMAGE_SIZE, 3], tf.float32, 'input')

    def build_graph(self, image):
        image = image / 128.0
        assert tf.test.is_gpu_available()
        image = tf.transpose(image, [0, 3, 1, 2])

        with argscope([Conv2DQuant, MaxPooling, BatchNorm], data_format="NCHW"), \
            argscope(Conv2DQuant, nl=tf.identity,
                     use_bias=False,
                     kernel_shape=3,
                     W_init=variance_scaling_initializer(mode='FAN_IN'),
                     nbit=self.qw,
                     is_quant=True if self.qw > 0 else False):
            layer = "..."

            layer = "..."

            layer = "..."

            # layer related lines...

            # cost related lines...

            return self.cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.02, trainable=False)
        # choose which optimizer?
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU

    logger.set_logger_dir(
        os.path.join('train_log', config.LOG_DIR))

    dataset_train = get_data('train')
    dataset_test = get_data('test')

    train_config = TrainConfig(
        model=Model(qw=config.QW, qa=config.QA),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong_vector')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.02), (80, 0.002), (160, 0.0002), (300, 0.00002)])
        ],
        max_epoch=config.MAX_EPOCH,
        nr_tower=max(get_num_gpu(), 1),
        session_init=SaverRestore(args.load) if args.load else None
    )
    num_gpu = max(get_num_gpu(), 1)
    launch_train_with_config(train_config, SyncMultiGPUTrainerParameterServer(num_gpu))













