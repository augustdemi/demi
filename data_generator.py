""" Code for loading data. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.python.platform import flags
from maml_temp.utils import get_images2
from maml_temp.vae_model import VAE
import EmoData as ED

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
        # data that is pre-resized using PIL with lanczos filter
        data_folder = config.get('data_folder', '/home/ml1323/project/robert_data/DISFA/kshot/0')
        subjects = os.listdir(data_folder)
        subjects.sort()
        subject_folders = [os.path.join(data_folder, subject) for subject in subjects]
        # random.seed(1)
        # random.shuffle(subject_folders)
        num_val = 0
        num_train = config.get('num_train', 14) - num_val
        self.metatrain_character_folders = subject_folders[:num_train]
        if FLAGS.test_set: # In test, runs only one test task for the entered subject
            self.metaval_character_folders = [subject_folders[FLAGS.subject_idx]]
            # self.metaval_character_folders = subject_folders[FLAGS.subject_idx:]
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
            print("folders: ",folders)
            num_total_batches = 600

        # make list of files
        print('Generating filenames')
        all_filenames = []
        for sub_folder in folders: # 쓰일 task수만큼만 경로 만든다. 이 task들이 iteration동안 어차피 반복될거니까
            # random.shuffle(sampled_character_folders)
            labels_and_images = get_images2(sub_folder, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False)
            # make sure the above isn't randomized order
            labels = [li[0] for li in labels_and_images] # 0 0 1 1 = on on off off
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)


        #################################################################################
        import cv2

        imgs = []
        for filename in all_filenames:
            img = cv2.imread(filename)
            imgs.append(img)

        pp = ED.image_pipeline.FACE_pipeline(
            histogram_normalization=True,
            grayscale=True,
            resize=True,
            rotation_range=3,
            width_shift_range=0.03,
            height_shift_range=0.03,
            zoom_range=0.03,
            random_flip=True,
        )

        img_arr, pts, pts_raw = pp.batch_transform(imgs, preprocessing=True, augmentation=False)
        vae_model = VAE(img_arr.shape[1:], (1, self.num_classes))


        nk = self.num_classes * FLAGS.update_batch_size
        img_arr = np.reshape(img_arr, [int(nk), int(len(img_arr)/nk)]) # len(img_arr)/nk = 2 * num of task

        z_arr = []
        for img_batch in img_arr:
            weights, z = vae_model.computeLatentVal(img_batch)
            z_arr.append(z)
        z_arr = np.concatenate(z_arr)
        self.pred_weights = weights
        #################################################################################

        # make queue for tensorflow to read from
        z_tensor = tf.convert_to_tensor(z_arr)
        examples_per_batch = self.num_classes * self.num_samples_per_class # 2NK = number of examples per task
        print(len(all_filenames))
        print(len(all_filenames)/examples_per_batch)
        print(self.batch_size)
        print(all_filenames)

        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape')
        for i in range(self.batch_size): #  batch_size = number of task
            image_batch = z_tensor[i*examples_per_batch:(i+1)*examples_per_batch]

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
        # all_label_batches = tf.reshape(all_label_batches, [int(all_label_batches.shape[0]), int(all_label_batches.shape[1]),1])
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches

