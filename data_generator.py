""" Code for loading data. """
import numpy as np
import os
import random
import tensorflow as tf
import cv2

from tensorflow.python.platform import flags
from utils import get_images
from vae_model import VAE
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
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = FLAGS.num_classes
        self.img_size = config.get('img_size', (160, 240))
        self.dim_input = np.prod(self.img_size)
        data_folder = FLAGS.datadir
        val_folder = FLAGS.valdir
        subjects = os.listdir(data_folder)
        subjects.sort()
        subject_folders = [os.path.join(data_folder, subject) for subject in subjects]
        num_val = 0
        num_train = FLAGS.meta_batch_size
        self.metatrain_character_folders = subject_folders[FLAGS.train_start_idx:FLAGS.train_start_idx + num_train]
        if FLAGS.test_set: # In test, runs only one test task for the entered subject
            self.metaval_character_folders = [subject_folders[FLAGS.subject_idx]]


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
        inputa_files = []
        inputb_files = []
        labelas = []
        labelbs = []
        # To have totally different inputa and inputb, they should be sampled at the same time and then splitted.
        for sub_folder in folders: # 쓰일 task수만큼만 경로 만든다. 이 task들이 iteration동안 어차피 반복될거니까
            # random.shuffle(sampled_character_folders)
            labels_and_images = get_images(sub_folder, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=True)
            # make sure the above isn't randomized order
            labels = [li[0] for li in labels_and_images] # 0 0 1 1 = off off on on
            filenames = [li[1] for li in labels_and_images]
            #Split data into a/b
            k = int(self.num_samples_per_class / 2) # = FLAGS.update_batch_size
            filenames = np.array(filenames).reshape(self.num_classes, self.num_samples_per_class)
            for files_per_class in filenames:
                inputa_files.extend(files_per_class[:k])
                inputb_files.extend(files_per_class[k:])

            labels = np.array(labels).reshape(self.num_classes, self.num_samples_per_class)
            for labels_per_class in labels:
                labelas.extend(labels_per_class[:k])
                labelbs.extend(labels_per_class[k:])

            all_filenames.extend(filenames) # just for debugging

        print("all_filenames: ", all_filenames)
        #################################################################################


        vae_model = VAE((self.img_size[0], self.img_size[1], 1), (1, self.num_classes))
        # inputa_files has (n*k * num_of_task) files.
        # make it to batch of which size is (n*k) : thus, the total number of batch = num_of_task
        batch_size = int(self.num_classes * FLAGS.update_batch_size)
        N_batch = num_of_task = int(len(inputa_files) / batch_size)  # len(inputa_files)/nk = num of task

        def latent_feature(file_names):
            file_names_batch = np.reshape(file_names, [N_batch, batch_size])

            z_arr = []
            for file_bath in file_names_batch:
                imgs = []
                for filename in file_bath:
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

                weights, z = vae_model.computeLatentVal(img_arr)
                z_arr.append(z)
            return np.concatenate(z_arr), weights

        inputa_latent_feat, self.pred_weights = latent_feature(inputa_files)
        inputb_latent_feat, self.pred_weights = latent_feature(inputb_files)
        #################################################################################

        print("original inputa: ", inputa_latent_feat)
        print("original inputb: ", inputb_latent_feat)
        np.random.seed(1)
        np.random.shuffle(inputa_latent_feat)
        np.random.seed(1)
        np.random.shuffle(labelas)
        np.random.seed(2)
        np.random.shuffle(inputb_latent_feat)
        np.random.seed(2)
        np.random.shuffle(labelbs)
        inputa_latent_feat_tensor = tf.convert_to_tensor(inputa_latent_feat)
        inputa_latent_feat_tensor = tf.reshape(inputa_latent_feat_tensor, [num_of_task, self.num_classes*k, 2000])
        inputb_latent_feat_tensor = tf.convert_to_tensor(inputb_latent_feat)
        inputb_latent_feat_tensor = tf.reshape(inputb_latent_feat_tensor, [num_of_task, self.num_classes*k, 2000])

        labelas_tensor = tf.convert_to_tensor(labelas)
        labelbs_tensor = tf.convert_to_tensor(labelbs)
        labelas_tensor = tf.one_hot(labelas_tensor, self.num_classes) ## (num_of_tast, 2NK, N)
        labelbs_tensor = tf.one_hot(labelbs_tensor, self.num_classes) ## (num_of_tast, 2NK, N)
        labelas_tensor = tf.reshape(labelas_tensor, [num_of_task, self.num_classes*k, 2])
        labelbs_tensor = tf.reshape(labelbs_tensor, [num_of_task, self.num_classes*k, 2])

        return inputa_latent_feat_tensor,inputb_latent_feat_tensor, labelas_tensor, labelbs_tensor

