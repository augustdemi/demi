""" Utility functions. """
import math
import os
import random
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Image helper
def get_images(path, label_int, seed, nb_samples=None, validate=False):
    print("============================================")
    print(">>>>>>>>>>>>>subject: ", path)
    labels = ['off', 'on']  # off = 0, on =1

    # check count of existing samples
    num_existing_samples = []
    for label in labels:
        img_path_list = os.listdir(os.path.join(path, label))
        num_existing_samples.append(len(img_path_list))

    # make the balance
    num_samples_to_select = [nb_samples, nb_samples]
    if num_existing_samples[0] < nb_samples:
        n_off_samples = 2 * math.floor(num_existing_samples[0] / 2)
        n_on_samples = 2 * nb_samples - n_off_samples
        num_samples_to_select = [n_off_samples, n_on_samples]
    elif num_existing_samples[1] < nb_samples:
        n_on_samples = 2 * math.floor(num_existing_samples[1] / 2)
        n_off_samples = 2 * nb_samples - n_on_samples
        num_samples_to_select = [n_off_samples, n_on_samples]
    print('num_samples_to_select: ', num_samples_to_select)

    def sampler(path, label, n_samples):
        print("-------------------------------")
        print("validate: ", validate)
        print("label: ", label)

        img_path_list = os.listdir(os.path.join(path, label))
        if validate:
            random.seed(1)
        else:
            random.seed(0)
        random_imgs = random.sample(img_path_list, n_samples)
        random_img_path = [os.path.join(path, label, img) for img in random_imgs]
        return random_img_path

    #각 task별로 k*2개 씩의 label 과 img담게됨. path = till subject.
    off_images = sampler(path, labels[0], num_samples_to_select[0])
    print('num of off_images: ', len(off_images))
    on_images = sampler(path, labels[1], num_samples_to_select[1])
    print('num of on_images: ', len(on_images))
    return off_images, on_images


def get_kshot_from_img_path(path, seed, nb_samples=None, validate=False):
    subject = int(path[-1])
    print("============================================")
    print(">>>>>>>>>>>>>subject: ", subject)
    # random seed는 subject에 따라서만 다르도록. 즉, 한 subject내에서는 k가 증가해도 계속 동일한 seed인것.
    labels = ['off', 'on']  # off = 0, on =1

    # check count of existing samples
    num_existing_samples = []
    for label in labels:
        img_path_list = open(path + label + '/file_path.csv').readline().split(',')
        num_existing_samples.append(len(img_path_list))

    # make the balance
    num_samples_to_select = [nb_samples, nb_samples]
    if num_existing_samples[0] < nb_samples:
        n_off_samples = 2 * math.floor(num_existing_samples[0] / 2)
        n_on_samples = 2 * nb_samples - n_off_samples
        num_samples_to_select = [n_off_samples, n_on_samples]
    elif num_existing_samples[1] < nb_samples:
        n_on_samples = 2 * math.floor(num_existing_samples[1] / 2)
        n_off_samples = 2 * nb_samples - n_on_samples
        num_samples_to_select = [n_off_samples, n_on_samples]
    print('num_samples_to_select: ', num_samples_to_select)

    def sampler(path, label, n_samples):
        print("-------------------------------")
        print("validate: ", validate)
        print("label: ", label)

        img_path_list = os.listdir(os.path.join(path, label))
        if validate:
            random.seed(subject + seed + 10)
        else:
            random.seed(subject + seed)
        random_imgs = random.sample(img_path_list, n_samples)
        random_img_path = [os.path.join(path, label, img) for img in random_imgs]
        return random_img_path

    # 각 task별로 k*2개 씩의 label 과 img담게됨. path = till subject.
    off_images = sampler(path, labels[0], num_samples_to_select[0])
    print('num of off_images: ', len(off_images))
    on_images = sampler(path, labels[1], num_samples_to_select[1])
    print('num of on_images: ', len(on_images))
    return off_images, on_images


def get_kshot_feature(kshot_path, feat_path, seed, nb_samples=None, validate=False):
    subject = kshot_path.split('/')[-1]
    print("============================================")
    print('kshot_path: ', kshot_path)
    print("subject: ", subject)
    # random seed는 subject에 따라서만 다르도록. 즉, 한 subject내에서는 k가 증가해도 계속 동일한 seed인것.
    labels = ['off', 'on']  # off = 0, on =1

    # check count of existing samples per off / on
    frames_n_features = []
    for label in labels:
        # 모든 feature 파일이 존재하는 경로
        feature_file_path = feat_path + '/' + subject + '.csv'
        print('feature_file_path: ', feature_file_path)
        f = open(feature_file_path, 'r')
        lines = f.readlines()
        all_feat_data = {}  # 모든 feature를 frame 을 key값으로 하여 dic에 저장해둠
        for line in lines:
            line = line.split(',')
            all_feat_data.update({line[1]: line[2:]})  # key = frame, value = feature vector


        # on/off 이미지를 구분해 놓은 csv파일로부터 라벨별 이미지 경로 읽어와( au, subject별 on 혹은 off 이미지)
        img_path_list = open(kshot_path + '/' + label + '/file_path.csv').readline().split(',')
        frame_n_feature_per_label = []
        try:
            for path in img_path_list:
                frame = path.split('/')[-1].split('.')[0].split('_')[0]
                frame_n_feature_per_label.append((frame, all_feat_data[frame]))
        except:
            print('CHECK DATA FOR LABEL: ', label, ' - ', img_path_list)
        frames_n_features.append(frame_n_feature_per_label)

    print('total off / on: ', len(frames_n_features[0]), len(frames_n_features[1]))
    # make the balance
    num_samples_to_select = [nb_samples, nb_samples]
    if len(frames_n_features[0]) < nb_samples:
        n_off_samples = 2 * math.floor(len(frames_n_features[0]) / 2)
        n_on_samples = 2 * nb_samples - n_off_samples
        num_samples_to_select = [n_off_samples, n_on_samples]
    elif len(frames_n_features[1]) < nb_samples:
        n_on_samples = 2 * math.floor(len(frames_n_features[1]) / 2)
        n_off_samples = 2 * nb_samples - n_on_samples
        num_samples_to_select = [n_off_samples, n_on_samples]
    print('num_samples_to_select: ', num_samples_to_select)

    def sampler(frames_n_features, n_samples):
        if validate:
            random.seed(1)
        else:
            random.seed(0)
        random_frames_n_features = random.sample(frames_n_features, n_samples)
        return random_frames_n_features

    # 각 task별로 k*2개 씩의 label 과 img담게됨. path = till subject.
    off_random_frames_n_features = sampler(frames_n_features[0], num_samples_to_select[0])
    print("-------------------------------")
    print("validate: ", validate)
    print("label: ", labels[0])
    print('num of off_images: ', len(off_random_frames_n_features))
    print('off_frames: ', [elt[0] for elt in off_random_frames_n_features])
    on_random_frames_n_features = sampler(frames_n_features[1], num_samples_to_select[1])
    print("-------------------------------")
    print("validate: ", validate)
    print("label: ", labels[1])
    print('num of on_images: ', len(on_random_frames_n_features))
    print('on_frames: ', [elt[0] for elt in on_random_frames_n_features])
    return [elt[1] for elt in off_random_frames_n_features], [elt[1] for elt in on_random_frames_n_features]


def get_kshot_feature_w_all_labels(kshot_path, feat_path, sampling_seed, nb_samples=None, check_sample=False):
    aus = ['au1', 'au2', 'au4', 'au6', 'au9', 'au12', 'au25', 'au26']

    subject = kshot_path.split('/')[-1]
    # print("============================================")
    # print("sampling seed: ", sampling_seed)
    # print('kshot_path: ', kshot_path)
    # print("subject: ", subject)
    # random seed는 subject에 따라서만 다르도록. 즉, 한 subject내에서는 k가 증가해도 계속 동일한 seed인것.
    labels = ['off', 'on']  # off = 0, on =1

    # check count of existing samples per off / on
    frames_n_features = []
    for label in labels:
        # 모든 feature 파일이 존재하는 경로
        feature_file_path = feat_path + '/' + subject + '.csv'
        f = open(feature_file_path, 'r')
        lines = f.readlines()
        all_feat_data = {}  # 모든 feature를 frame 을 key값으로 하여 dic에 저장해둠
        for line in lines:
            line = line.split(',')
            all_feat_data.update({line[1]: line[2:]})  # key = frame, value = feature vector

        # on/off 이미지를 구분해 놓은 csv파일로부터 라벨별 이미지 경로 읽어와( au, subject별 on 혹은 off 이미지)
        img_path_list = open(kshot_path + '/' + label + '/file_path.csv').readline().split(',')
        frame_n_feature_per_label = []
        try:
            for path in img_path_list:
                frame = path.split('/')[-1].split('.')[0].split('_')[0]
                frame_n_feature_per_label.append((frame, all_feat_data[frame]))
        except:
            print('CHECK DATA FOR LABEL: ', kshot_path, label, ' - ', img_path_list)
        frames_n_features.append(frame_n_feature_per_label)

    # print('total off / on: ', len(frames_n_features[0]), len(frames_n_features[1]))
    # make the balance
    num_samples_to_select = [nb_samples, nb_samples]
    if len(frames_n_features[0]) < nb_samples:
        n_off_samples = 2 * math.floor(len(frames_n_features[0]) / 2)
        n_on_samples = 2 * nb_samples - n_off_samples
        num_samples_to_select = [n_off_samples, n_on_samples]
    elif len(frames_n_features[1]) < nb_samples:
        n_on_samples = 2 * math.floor(len(frames_n_features[1]) / 2)
        n_off_samples = 2 * nb_samples - n_on_samples
        num_samples_to_select = [n_off_samples, n_on_samples]

    # print('num_samples_to_select: ', num_samples_to_select)

    def sampler(frames_n_features, n_samples):
        random.seed(sampling_seed)
        random_frames_n_features = random.sample(frames_n_features, n_samples)
        return random_frames_n_features

    # 각 task별로 k*2개 씩의 label 과 img담게됨. path = till subject.
    off_random_frames_n_features = sampler(frames_n_features[0], num_samples_to_select[0])
    # print("-------------------------------")
    # print("label: ", labels[0])
    # print('num of off_images: ', len(off_random_frames_n_features))
    # print('off_frames: ', [elt[0] for elt in off_random_frames_n_features])
    on_random_frames_n_features = sampler(frames_n_features[1], num_samples_to_select[1])
    # print("-------------------------------")
    # print("label: ", labels[1])
    # print('num of on_images: ', len(on_random_frames_n_features))
    # print('on_frames: ', [elt[0] for elt in on_random_frames_n_features])

    if check_sample:
        import pickle
        out = open(
            '/home/ml1323/project/robert_code/new/check_labels/' + subject + '_' + kshot_path.split('/')[-2] + ".pkl",
            'wb')
        pickle.dump({'off': [elt[0] for elt in off_random_frames_n_features],
                     'on': [elt[0] for elt in on_random_frames_n_features]}, out, protocol=2)

    off_frames_idx = [int(elt[0].split('frame')[1]) for elt in off_random_frames_n_features]
    on_frames_idx = [int(elt[0].split('frame')[1]) for elt in on_random_frames_n_features]
    on_sample_labels = []
    off_sample_labels = []
    binary_intensity = lambda lab: 1 if lab > 0 else 0
    for au in aus:
        # label_path = os.path.join('./data/label', subject, subject + '_' + au + '.txt')
        label_path = os.path.join('/home/ml1323/project/robert_data/DISFA/label', subject, subject + '_' + au + '.txt')
        with open(label_path) as f:
            lines = np.array(f.readlines())
            selected_labels_for_on = [binary_intensity(int(line.split(",")[1].split("\n")[0])) for line in lines[on_frames_idx]]
            selected_labels_for_off = [binary_intensity(int(line.split(",")[1].split("\n")[0])) for line in lines[off_frames_idx]]
            on_sample_labels.append(selected_labels_for_on)
            off_sample_labels.append(selected_labels_for_off)
    on_sample_labels = np.array(on_sample_labels).transpose(1,0)
    off_sample_labels = np.array(off_sample_labels).transpose(1,0)
    #
    # print("------on_sample_labels:")
    # print(on_sample_labels)
    # print("------off_sample_labels:")
    # print(off_sample_labels)

    return [elt[1] for elt in off_random_frames_n_features], [elt[1] for elt in
                                                              on_random_frames_n_features], off_sample_labels, on_sample_labels


def get_all_feature_w_all_labels(feature_files, label_paths, test_split_seed=-1):
    from feature_layers import feature_layer

    all_subject_features = []
    print(">>>>>>>>>>>>>>>>> embedding model: ", FLAGS.vae_model)
    three_layers = feature_layer(10, FLAGS.num_au)
    three_layers.loadWeight(FLAGS.vae_model)

    print('---- get feature vec')
    existing_frames_in_feat_vec = []
    for feature_file in feature_files:
        print("subject: ", feature_file.split('/')[-1])
        with open(feature_file, 'r') as f:
            lines = f.readlines()
            one_subject_features = [np.array(range(2048)) for _ in
                                    range(4845)]  # 모든 feature를 frame 을 key값으로 하여 dic에 저장해둠
            for line in lines:
                line = line.split(',')
                frame_idx = int(line[1].split('frame')[1])
                existing_frames_in_feat_vec.append(frame_idx)
                feat_vec = np.array([float(elt) for elt in line[2:]])
                if frame_idx < 4845:
                    one_subject_features[frame_idx] = feat_vec  # key = frame, value = feature vector
            one_subject_features = np.array(one_subject_features)
            one_subject_features = three_layers.model_final_latent_feat.predict(one_subject_features)
            all_subject_features.append(one_subject_features)

    print('---- get label')
    test_b_frame = []
    if test_split_seed > -1:
        random.seed(test_split_seed)
        test_a_frame = random.sample(existing_frames_in_feat_vec,
                                     int(len(existing_frames_in_feat_vec) / 2))
        test_b_frame = [i for i in existing_frames_in_feat_vec if i not in test_a_frame]
        existing_frames_in_feat_vec = test_a_frame

    binary_intensity = lambda lab: 1 if lab > 0 else 0
    aus = ['au1', 'au2', 'au4', 'au6', 'au9', 'au12', 'au25', 'au26']
    all_subject_labels = []  # list of feat_vec array per subject
    all_subject_on_intensity_info = []
    all_subject_off_intensity_info = []
    for label_path in label_paths:
        subject = label_path.split('/')[-1]
        print(subject)
        all_labels_per_subj = []
        all_au_on_intensity_info = []
        all_au_off_intensity_info = []
        for au in aus:
            f = open(os.path.join(label_path, subject + '_' + au + '.txt'), 'r')
            lines = f.readlines()[:4845]
            labels_per_subj_per_au = [binary_intensity(np.float32(line.split(',')[1].split('\n')[0])) for line in lines]
            if len(labels_per_subj_per_au) < 4845:
                labels_per_subj_per_au.append(-1)
            all_labels_per_subj.append(labels_per_subj_per_au)

            # for dataframe
            on_intensity_info = [i for i in range(len(lines)) if
                                 (float(lines[i].split(',')[1].split('\n')[0]) > 0) and (
                                 i in existing_frames_in_feat_vec)]
            off_intensity_info = [i for i in range(4845) if
                                  (i not in on_intensity_info) and (i in existing_frames_in_feat_vec)]
            all_au_on_intensity_info.append(on_intensity_info)
            all_au_off_intensity_info.append(off_intensity_info)
        all_labels_per_subj = np.transpose(np.array(all_labels_per_subj), (1, 0))
        all_subject_labels.append(all_labels_per_subj)
        # for dataframe
        all_subject_on_intensity_info.append(all_au_on_intensity_info)
        all_subject_off_intensity_info.append(all_au_off_intensity_info)
    all_subject_features = np.array(all_subject_features)
    all_subject_labels = np.array(all_subject_labels)
    print('===================')
    print("--- z_arr len:", len(all_subject_features[0][0]))
    print('all_subject_features: ', all_subject_features.shape)
    print('all_subject_labels: ', all_subject_labels.shape)
    all_subject_on_intensity_info = np.array(all_subject_on_intensity_info)  # 14*8
    all_subject_off_intensity_info = np.array(all_subject_off_intensity_info)

    import pandas as pd
    on_pd_data = {}
    for i in range(len(aus)):
        on_pd_data[aus[i]] = all_subject_on_intensity_info[:, i]
    on_df = pd.DataFrame(
        data=on_pd_data
    )

    off_pd_data = {}
    for i in range(len(aus)):
        off_pd_data[aus[i]] = all_subject_off_intensity_info[:, i]
    off_df = pd.DataFrame(
        data=off_pd_data
    )

    return all_subject_features, all_subject_labels, on_df, off_df, test_b_frame



def test(kshot_path, feat_path, label_path, sampling_seed, subject, nb_samples):
    print(">>>>>>>>>>>>>>>>> embedding model: ", FLAGS.vae_model)
    print(">>>>>>>>>>>>>>>>> feat_path: ", feat_path)
    print(">>>>>>>>>>>>>>>>> kshot_path: ", kshot_path)
    print(">>>>>>>>>>>>>>>>> subject: ", subject)
    from feature_layers import feature_layer
    three_layers = feature_layer(10, FLAGS.num_au)
    three_layers.loadWeight(FLAGS.vae_model, FLAGS.au_idx, num_au_for_rm=FLAGS.num_au)
    aus = ['au1', 'au2', 'au4', 'au6', 'au9', 'au12', 'au25', 'au26']

    #### save all label info ####
    labels = ['off', 'on']  # off = 0, on =1
    binary_intensity = lambda lab: 1 if lab > 0 else 0
    all_label_info_per_subj = []
    for au in aus:
        with open(os.path.join(label_path, subject + '_' + au + '.txt'), 'r') as f:
            lines = f.readlines()[:4845]
            labels_per_subj_per_au = [binary_intensity(float(line.split(',')[1].split('\n')[0])) for line in lines]
            all_label_info_per_subj.append(labels_per_subj_per_au)
    all_label_info_per_subj = np.transpose(np.array(all_label_info_per_subj), (1, 0))

    #### save all resnet feat ####
    f = open(feat_path, 'r')
    lines = f.readlines()
    all_feat_data = {}  # 모든 feature를 frame 을 key값으로 하여 dic에 저장해둠
    for line in lines:
        line = line.split(',')
        all_feat_data.update({int(line[1].split('frame')[1]): line[2:]})  # key = frame, value = feature vector

    # for each au, save on/off frames of this subject
    all_aus_feat_vec = []
    all_aus_labels = []
    for au in aus:
        frames_n_features_per_au = []
        for label in labels:
            # on/off 이미지를 구분해 놓은 csv파일로부터 라벨별 이미지 경로 읽어와( au, subject별 on 혹은 off 이미지)
            img_path_list = open(os.path.join(kshot_path, au, subject, label, 'file_path.csv')).readline().split(',')
            frame_n_feature_per_label = []
            try:
                for path in img_path_list:
                    frame = int(path.split('/')[-1].split('.')[0].split('_')[0].split('frame')[1])
                    if frame < 4845:
                        frame_n_feature_per_label.append((frame, all_feat_data[frame]))
            except:
                print('CHECK DATA FOR LABEL: ', kshot_path, label, ' - ', img_path_list)
            frames_n_features_per_au.append(frame_n_feature_per_label)

        print('total off / on of ', au, ': ', len(frames_n_features_per_au[0]), len(frames_n_features_per_au[1]))
        # make the balance
        num_samples_to_select = [nb_samples, nb_samples]
        num_samples_to_select[1] = min(nb_samples, len(frames_n_features_per_au[1]))
        num_samples_to_select[0] = 2 * nb_samples - num_samples_to_select[1]

        def sampler(frames_n_features, n_samples):
            random.seed(sampling_seed)
            random_frames_n_features = random.sample(frames_n_features, n_samples)
            return random_frames_n_features

        # 각 task별로 k*2개 씩의 label 과 img담게됨. path = till subject.
        off_random_frames_n_features = sampler(frames_n_features_per_au[0], num_samples_to_select[0])
        print("-------------------------------")
        print(au)
        print("-------------------------------")
        print("label: ", labels[0])
        print('num of off_images: ', len(off_random_frames_n_features))
        print('off_frames: ', [elt[0] for elt in off_random_frames_n_features])
        on_random_frames_n_features = sampler(frames_n_features_per_au[1], num_samples_to_select[1])
        print("-------------------------------")
        print("label: ", labels[1])
        print('num of on_images: ', len(on_random_frames_n_features))
        print('on_frames: ', [elt[0] for elt in on_random_frames_n_features])

        # get label of this au
        selected_frame_idx = [elt[0] for elt in off_random_frames_n_features]  # put off_idx
        selected_frame_idx.extend([elt[0] for elt in on_random_frames_n_features])  # put on_idx
        random.seed(0)
        random.shuffle(selected_frame_idx)
        all_aus_labels.append(all_label_info_per_subj[selected_frame_idx])
        # get feature vector of this au
        feat_vec = [elt[1] for elt in off_random_frames_n_features]  # put off_features
        feat_vec.extend([elt[1] for elt in on_random_frames_n_features])  # put on_features
        random.seed(0)
        random.shuffle(feat_vec)
        feat_vec = three_layers.model_final_latent_feat.predict(feat_vec)
        all_aus_feat_vec.append(feat_vec)

    return np.array(all_aus_feat_vec), np.array(all_aus_labels), all_label_info_per_subj



## Network helpers
def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID', residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    if FLAGS.max_pool:
        conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    else:
        conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope)
    if FLAGS.max_pool:
        normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
    return normed

def normalize(inp, activation, reuse, scope):
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.square(pred - label) / FLAGS.update_batch_size

def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size


def xent_sig(pred, label):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size


def scaling(data):
    return -1 + 2 * data
