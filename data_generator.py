""" Code for loading data. """
import numpy as np
import os
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

    def __init__(self, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
        """
        self.num_samples_per_class = FLAGS.update_batch_size * 2
        self.num_classes = FLAGS.num_classes
        self.img_size = config.get('img_size', (160, 240))
        self.dim_input = np.prod(self.img_size)
        data_folder = FLAGS.datadir
        subjects = os.listdir(data_folder)
        subjects.sort()
        subject_folders = [os.path.join(data_folder, subject) for subject in subjects]
        self.metatrain_character_folders = subject_folders[
                                           FLAGS.sbjt_start_idx:FLAGS.sbjt_start_idx + FLAGS.meta_batch_size]
        if FLAGS.test_set:  # In test, runs only one test task for the entered subject
            self.metatest_character_folders = [subject_folders[FLAGS.subject_idx]]
        else:
            if FLAGS.train_test:  # test task로 다시한번 모델을 retrain할때는 같은 test task중 train에 쓰이지 않은 다른 샘플을 선택하여 validate
                self.metatest_character_folders = self.metatrain_character_folders
            else:
                self.metatest_character_folders = self.metatrain_character_folders

    def make_data_tensor(self, train=True):
        if train:
            print("===================================2")
            folders = self.metatrain_character_folders
            print(">>>>>>> train folders: ", folders)
        else:
            folders = self.metatest_character_folders
            print(">>>>>>> test folders: ", folders)

        # make list of files
        print('Generating filenames')
        inputa_files = []
        inputb_files = []
        labelas = []
        labelbs = []
        # To have totally different inputa and inputb, they should be sampled at the same time and then splitted.
        for sub_folder in folders:  # 쓰일 task수만큼만 경로 만든다. 이 task들이 iteration동안 어차피 반복될거니까
            # random.shuffle(sampled_character_folders)
            if train:
                off_imgs, on_imgs = get_images(sub_folder, range(self.num_classes), FLAGS.kshot_seed,
                                               nb_samples=FLAGS.update_batch_size * 2, validate=False)
            else:
                off_imgs, on_imgs = get_images(sub_folder, range(self.num_classes), FLAGS.kshot_seed,
                                               nb_samples=FLAGS.update_batch_size * 2, validate=True)
            # Split data into a/b
            half_off_img = int(len(off_imgs) / 2)
            half_on_img = int(len(on_imgs) / 2)
            for i in range(half_off_img):
                inputa_files.append(off_imgs[2 * i])
                inputb_files.append(off_imgs[2 * i + 1])
            for i in range(half_on_img):
                inputa_files.append(on_imgs[2 * i])
                inputb_files.append(on_imgs[2 * i + 1])
            label_for_this_subj = [0] * half_off_img
            label_for_this_subj.extend([1] * half_on_img)
            labelas.extend(label_for_this_subj)
            labelbs.extend(label_for_this_subj)


        print(">>>> inputa_files: ", inputa_files)
        print("--------------------------------------------")
        print(">>>> inputb_files: ", inputb_files)
        print(">>> labelas: ", labelas)
        print(">>>>>>>>>>>>>>>>> vae_model: ", FLAGS.vae_model)
        print(">>>>>>>>>>>>>>>>>> random seed for kshot: ", FLAGS.kshot_seed)
        print(">>>>>>>>>>>>>>>>>> random seed for weight: ", FLAGS.weight_seed)

        #################################################################################



        batch_size = 10
        vae_model = VAE((self.img_size[0], self.img_size[1], 1), batch_size, FLAGS.num_au)
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

        def latent_feature(file_names):
            nb_samples = len(file_names)
            t0, t1 = 0, batch_size
            z_arr = []
            while True:
                t1 = min(nb_samples, t1)
                file_names_batch = file_names[t0:t1]
                imgs = [cv2.imread(filename) for filename in file_names_batch]
                img_arr, pts, pts_raw = pp.batch_transform(imgs, preprocessing=True, augmentation=False)
                weights, z = vae_model.computeLatentVal(img_arr, FLAGS.vae_model, FLAGS.au_idx)
                z_arr.append(z)
                if t1 == nb_samples: break
                t0 += batch_size  # 작업한 배치 사이즈만큼 t0와 t1늘림
                t1 += batch_size

            return np.concatenate(z_arr), weights

        inputa_latent_feat, self.pred_weights = latent_feature(inputa_files)
        inputb_latent_feat, self.pred_weights = latent_feature(inputb_files)
        print(">>> z_arr len:", len(inputa_latent_feat))
        #################################################################################

        # print("original inputa: ", inputa_latent_feat)
        # print("original inputb: ", inputb_latent_feat)
        np.random.seed(1)
        np.random.shuffle(inputa_latent_feat)
        np.random.seed(1)
        np.random.shuffle(labelas)
        np.random.seed(2)
        np.random.shuffle(inputb_latent_feat)
        np.random.seed(2)
        np.random.shuffle(labelbs)
        inputa_latent_feat_tensor = tf.convert_to_tensor(inputa_latent_feat)
        inputa_latent_feat_tensor = tf.reshape(inputa_latent_feat_tensor,
                                               [FLAGS.meta_batch_size, FLAGS.update_batch_size * 2, 2000])
        inputb_latent_feat_tensor = tf.convert_to_tensor(inputb_latent_feat)
        inputb_latent_feat_tensor = tf.reshape(inputb_latent_feat_tensor,
                                               [FLAGS.meta_batch_size, FLAGS.update_batch_size * 2, 2000])

        labelas_tensor = tf.convert_to_tensor(labelas)
        labelbs_tensor = tf.convert_to_tensor(labelbs)
        labelas_tensor = tf.one_hot(labelas_tensor, self.num_classes)  ## (num_of_tast, 2NK, N)
        labelbs_tensor = tf.one_hot(labelbs_tensor, self.num_classes)  ## (num_of_tast, 2NK, N)
        labelas_tensor = tf.reshape(labelas_tensor, [FLAGS.meta_batch_size, FLAGS.update_batch_size * 2, 2])
        labelbs_tensor = tf.reshape(labelbs_tensor, [FLAGS.meta_batch_size, FLAGS.update_batch_size * 2, 2])

        return inputa_latent_feat_tensor, inputb_latent_feat_tensor, labelas_tensor, labelbs_tensor
