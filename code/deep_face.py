
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf


from matplotlib import pyplot as plt
from os import path
from tensorflow import keras

IMAGE_SIZE = (152, 152)
CHANNELS = 3
NUM_CLASSES = 8631
# TPU_WORKER = 'grpc://10.0.0.1:8470'
BATCH_SIZE = 1024
LEARN_RATE = 0.01 * (BATCH_SIZE/128)
MOMENTUM = 0.9
EPOCHS = 15


def create_deepface(image_size=IMAGE_SIZE, channels=CHANNELS, num_classes=NUM_CLASSES, learn_rate=LEARN_RATE, momentum=MOMENTUM):
    """
    Deep CNN architecture primarily for Face Recognition,
    Face Verification and Face Representation (feature extraction) purposes
    "DeepFace: Closing the Gap to Human-Level Performance in Face Verification"
    CNN architecture proposed by Taigman et al. (CVPR 2014)
    """

    wt_init = keras.initializers.RandomNormal(mean=0, stddev=0.01)
    bias_init = keras.initializers.Constant(value=0.5)

    """
    Construct certain functions 
    for using some common parameters
    with network layers
    """
    def conv2d_layer(**args):
        return keras.layers.Conv2D(**args,
                                   kernel_initializer=wt_init,
                                   bias_initializer=bias_init,
                                   activation=keras.activations.relu)

    def lc2d_layer(**args):
        return keras.layers.LocallyConnected2D(**args,
                                               kernel_initializer=wt_init,
                                               bias_initializer=bias_init,
                                               activation=keras.activations.relu)

    def dense_layer(**args):
        return keras.layers.Dense(**args,
                                  kernel_initializer=wt_init,
                                  bias_initializer=bias_init)

    """
    Create the network using
    tf.keras.layers.Layer(s)
    """
    deepface = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(
            *image_size, channels), name='I0'),
        conv2d_layer(filters=32, kernel_size=11, name='C1'),
        keras.layers.MaxPooling2D(
            pool_size=3, strides=2, padding='same',  name='M2'),
        conv2d_layer(filters=16, kernel_size=9, name='C3'),
        lc2d_layer(filters=16, kernel_size=9, name='L4'),
        lc2d_layer(filters=16, kernel_size=7, strides=2, name='L5'),
        lc2d_layer(filters=16, kernel_size=5, name='L6'),
        keras.layers.Flatten(name='F0'),
        dense_layer(units=4096, activation=keras.activations.relu, name='F7'),
        keras.layers.Dropout(rate=0.5, name='D0'),
        dense_layer(units=num_classes,
                    activation=keras.activations.softmax, name='F8')
    ], name='DeepFace')
    deepface.summary()

    """
    A tf.keras.optimizers.SGD will
    be used for training,
    and compile the model
    """
    sgd_opt = keras.optimizers.SGD(lr=learn_rate, momentum=momentum)
    cce_loss = keras.losses.categorical_crossentropy

    deepface.compile(optimizer=sgd_opt, loss=cce_loss, metrics=['accuracy'])
    return deepface


DOWNLOAD_PATH = 'https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/VGGFace2_DeepFace_weights_val-0.9034.h5.zip'
MD5_HASH = '0b21fb70cd6901c96c19ac14c9ea8b89'


def get_weights():
    filename = 'deepface.zip'
    downloaded_file_path = keras.utils.get_file(filename, DOWNLOAD_PATH,
                                                md5_hash=MD5_HASH, extract=True)
    downloaded_h5_file = path.join(path.dirname(downloaded_file_path),
                                   path.basename(DOWNLOAD_PATH).rstrip('.zip'))
    return downloaded_h5_file


SHUFFLE_BUFFER = 1


class Dataset:
    @staticmethod
    def preprocess_image(img):
        img /= 255.0
        return img

    @staticmethod
    def get_class_labels(file_path):
        line_seperator = '\n'
        file_contents = tf.io.read_file(file_path)
        file_contents = tf.expand_dims(file_contents, axis=-1)

        class_labels = tf.strings.split(file_contents, sep=line_seperator)
        class_labels = class_labels.values[:-1]
        return class_labels

    def __init__(self, cl_path, dataset_path, image_size, batch_size, shuffle=True):
        self.dataset_path, self.image_size, self.batch_size, self.shuffle = [value for value in (
            dataset_path, image_size, batch_size, shuffle)]  # use shuffle only with train, not with test
        self.class_labels = Dataset.get_class_labels(cl_path)

        # with tf.Session() as sess:
        #   self.num_classes = sess.run(tf.shape(self.class_labels)[0])
        self.num_classes = 8631
        self.data = self.get_dataset()

    def get_image_and_class(self, image, classl):
        classl = tf.math.equal(self.class_labels, classl)
        classl = tf.cast(classl, tf.int32)
        classl = tf.argmax(classl, axis=-1)
        classl = tf.one_hot(classl, self.num_classes)

        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.compat.v1.image.resize_image_with_pad(
            image, self.image_size[0], self.image_size[1])
        image = tf.cast(image, tf.float32)
        image = Dataset.preprocess_image(image)

        return image, classl

    def read_tfrec(self, example):
        feature = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'class': tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(example, feature)
        return self.get_image_and_class(example['image'], example['class'])

    def get_dataset(self):
        cycle_length = 32
        prefetch_size = 1
        option = tf.data.Options()
        option.experimental_deterministic = False

        ds = tf.data.Dataset.list_files(self.dataset_path + '/*/*')
        ds = ds.with_options(option)
        ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=cycle_length,
                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.map(self.read_tfrec, tf.data.experimental.AUTOTUNE)
        if self.shuffle:
            ds = ds.shuffle(SHUFFLE_BUFFER)
        ds = ds.repeat()
        ds = ds.batch(self.batch_size, drop_remainder=True)
        return ds.prefetch(prefetch_size)

    def get_train_test_dataset(cl_path, dataset_path, image_size, batch_size):
        train_path = '/train'
        test_path = '/test'
        train, test = [
            Dataset(cl_path, dataset_path + curr_path,
                    image_size, batch_size, training)
            for curr_path, training in zip((train_path, test_path), (True, False))
        ]
        return train, test


CL_PATH = './class_labels.txt'
DATASET_PATH = './image'
TB_PATH = './image'

keras.backend.clear_session()
# TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']

...

# tpu_model = tf.contrib.tpu.keras_to_tpu_model(
# training_model,
# strategy=tf.contrib.tpu.TPUDistributionStrategy(
#     tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
# tpu_cluster = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=TPU_WORK)
# tf.contrib.distribute.initailize_tpu_syste,(tpu_cluster)
# strategy = tf.contrib.distribute.TPUStratedy(tpu_Cluster)


train, val = Dataset.get_train_test_dataset(
    CL_PATH, DATASET_PATH, IMAGE_SIZE, BATCH_SIZE)
train_samples, val_samples = 18, 1
Dataset.SHUFFLE_BUFFER = train_samples
# assert train.num_classes == val.num_classes == NUM_CLASSES

# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor =0.1, patience = 1, min_lr = 0.0001, verbose =1)
# tensorboard = keras.callback.TensorBoard(TB_PATH)
# checkpoints = keras.callbacks.ModelCheckpoint('weight.{epoch:02d}_{val_acc:.4f}.hdf5', monitor='val_acc', save_weights_only = True)
# cbs = [reduce_lr, checkpoints, tensorboard]

# with stratedy.scope():
model = create_deepface(IMAGE_SIZE, CHANNELS,
                        NUM_CLASSES, LEARN_RATE, MOMENTUM)

train_history = model.fit(train.data, steps_per_epoch=train_samples // BATCH_SIZE + 1,
                          validation_data=val.data, validation_steps=val_samples // BATCH_SIZE + 1, epochs=EPOCHS)
model.save('model.h5')


def save_plots():
    acc = train_history.histroy['acc']
    val_acc = train_history.history['val_acc']

    loss = train_history.history['loss']
    val_loss = train_histroy.histroy['val_loss']
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legent(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.xlabel('Epochs')
    plt.title('Loss')

    plt.savefig('epoch_wise_loss_acc.png')


save_plots()
