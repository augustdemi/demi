""" Code for loading data. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import get_images, get_images2

FLAGS = flags.FLAGS

class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, num_samples_per_class, batch_size, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)

        self.num_classes = config.get('num_classes', FLAGS.num_classes)
        self.img_size = config.get('img_size', (160, 240))
        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes
        # data that is pre-resized using PIL with lanczos filter
        data_folder = config.get('data_folder', '../data/1')

        subject_folders = [os.path.join(data_folder, subject) \
                             for subject in os.listdir(data_folder)]
        random.seed(1)
        random.shuffle(subject_folders)
        num_val = 3
        num_train = config.get('num_train', 14) - num_val
        self.metatrain_character_folders = subject_folders[:num_train]
        if FLAGS.test_set:
            self.metaval_character_folders = subject_folders[num_train + num_val:]
        else:
            self.metaval_character_folders = subject_folders[num_train:num_train + num_val]
        self.rotations = config.get('rotations', [0])


    def make_data_tensor(self, train=True):
        if train:
            folders = self.metatrain_character_folders
            # number of tasks, not number of meta-iterations. (divide by metabatch size to measure)
            num_total_batches = 200000
        else:
            folders = self.metaval_character_folders
            num_total_batches = 600

        # make list of files
        print('Generating filenames')
        all_filenames = []
        for sub_folder in folders: # 쓰일 task수만큼만 경로 만든다. 이 task들이 iteration동안 어차피 반복될거니까
            # random.shuffle(sampled_character_folders)
            labels_and_images = get_images2(sub_folder, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False)
            # make sure the above isn't randomized order
            labels = [li[0] for li in labels_and_images]
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)

        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        image = tf.image.decode_jpeg(image_file)
        image.set_shape((self.img_size[0], self.img_size[1], 1))
        image = tf.reshape(image, [self.dim_input])
        image = tf.cast(image, tf.float32) / 255.0
        num_preprocess_threads = 1 # TODO - enable this to be set to >1
        min_queue_examples = 256
        examples_per_batch = self.num_classes * self.num_samples_per_class
        batch_image_size = self.batch_size * examples_per_batch
        print('Batching images')
        images = tf.train.batch(
                [image],
                batch_size = batch_image_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_image_size,
                )



        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape')
        for i in range(self.batch_size):
            image_batch = images[i*examples_per_batch:(i+1)*examples_per_batch]

            label_batch = tf.convert_to_tensor(labels)
            new_list, new_label_list = [], []
            for k in range(self.num_samples_per_class):
                class_idxs = tf.range(0, self.num_classes)
                class_idxs = tf.random_shuffle(class_idxs)

                true_idxs = class_idxs*self.num_samples_per_class + k
                new_list.append(tf.gather(image_batch,true_idxs))

                new_label_list.append(tf.gather(label_batch, true_idxs))
            new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.reshape(all_label_batches, [int(all_label_batches.shape[0]), int(all_label_batches.shape[1]),1])
        # all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches

