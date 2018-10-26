import numpy as np
import tensorflow as tf

from EmoEstimator.utils.evaluate import print_summary
from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags
from datetime import datetime
import os
import pickle
from feature_layers import feature_layer
start_time = datetime.now()
FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'disfa', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 2, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 100,
                     'number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 1, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5,
                     'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.')  # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')

## Model options
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('datadir', '/home/ml1323/project/robert_data/DISFA/new_dataset/train/au0/', 'directory for data.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_integer('num_test_pts', 1, 'number of iteration to increase the test points')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('subject_idx', 0, 'subject index to test')
flags.DEFINE_integer('train_update_batch_size', -1,
                     'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1,
                   'value of inner gradient step step during training. (use if you want to test with a different value)')  # 0.1 for omniglot

flags.DEFINE_bool('init_weight', True, 'Initialize weights from the base model')
flags.DEFINE_bool('train_test', False, 're-train model with the test set')
flags.DEFINE_bool('train_test_inc', False, 're-train model increasingly')
# for train, train_test
flags.DEFINE_integer('sbjt_start_idx', 0, 'starting subject index')
# for test_test, test_train
flags.DEFINE_integer('num_test_tasks', 1, 'num of task for test')
flags.DEFINE_string('testset_dir', './data/1/', 'directory for test set')
flags.DEFINE_string('test_result_dir', 'robert', 'directory for test result log')
# for train_test, test_test
flags.DEFINE_string('keep_train_dir', None,
                    'directory to read already trained model when training the model again with test set')

flags.DEFINE_integer('kshot_seed', 0, 'seed for k shot sampling')
flags.DEFINE_integer('weight_seed', 0, 'seed for initial weight')
flags.DEFINE_integer('num_au', 8, 'number of AUs used to make AE')
flags.DEFINE_integer('num_au_to_test', 8, 'number of AUs to test')
flags.DEFINE_integer('au_idx', 8, 'au index to use in the given AE')
flags.DEFINE_string('vae_model', './model_au_12.h5', 'vae model dir from robert code')
flags.DEFINE_string('gpu', "0,1,2,3", 'vae model dir from robert code')
flags.DEFINE_bool('global_test', False, 'get test evaluation throughout all test tasks')
flags.DEFINE_bool('all_sub_model', True, 'model is trained with all train/test tasks')
flags.DEFINE_string('model', '', 'model name')
flags.DEFINE_bool('temp_train', False, 'test the test set with train-model')
flags.DEFINE_bool('local', False, 'save path from local weight')
flags.DEFINE_string('feature_path', "", 'path for feature vector')
flags.DEFINE_string('vae_model_to_test', '', 'vae model dir from robert code')


def test_each_subject(w, b, sbjt_start_idx):  # In case when test the model with the whole rest frames
    batch_size = 10
    three_layers = feature_layer(batch_size, FLAGS.num_au)
    print("!!!!!!!!!!!!!!!!!!")
    print(w.shape)
    print("!!!!!!!!!!!!!!!!!!")

    three_layers.loadWeight(FLAGS.vae_model, FLAGS.au_idx, num_au_for_rm=FLAGS.num_au, w=w, b=b)

    test_subjects = os.listdir(FLAGS.testset_dir)
    test_subjects.sort()

    test_subject = test_subjects[sbjt_start_idx]

    print("====================> subject: ", test_subject)
    data = pickle.load(open(FLAGS.testset_dir + test_subject, "rb"), encoding='latin1')
    test_features = data['test_features']
    y_hat = three_layers.model_intensity.predict(test_features)
    if FLAGS.au_idx < 8:
        lab = data['lab'][:, FLAGS.au_idx]
        y_lab = np.reshape(lab, (lab.shape[0], 1, lab.shape[1]))
    else:
        y_lab = data['lab']
    return y_hat, y_lab


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    TOTAL_NUM_AU = 8
    all_au = ['au1', 'au2', 'au4', 'au6', 'au9', 'au12', 'au25', 'au26']

    if not FLAGS.train:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1
        temp_kshot = FLAGS.update_batch_size
        FLAGS.update_batch_size = 1

    data_generator = DataGenerator()

    dim_output = data_generator.num_classes
    dim_input = data_generator.dim_input

    inputa, inputb, labela, labelb = data_generator.make_data_tensor(train=False)
    metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    model = MAML(dim_input, dim_output)
    model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=20)

    sess = tf.InteractiveSession()


    if not FLAGS.train:
        # change to original meta batch size when loading model.
        FLAGS.update_batch_size = temp_kshot
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    print('initial weights: ', sess.run('model/b1:0'))
    print("========================================================================================")

    ################## Test ##################
    def _load_weight_m(trained_model_dir):
        all_au = ['au1', 'au2', 'au4', 'au6', 'au9', 'au12', 'au25', 'au26']
        if FLAGS.au_idx < TOTAL_NUM_AU: all_au = [all_au[FLAGS.au_idx]]
        w_arr = None
        b_arr = None
        for au in all_au:
            model_file = None
            print('model file dir: ', FLAGS.logdir + '/' + au + '/' + trained_model_dir)
            model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + au + '/' + trained_model_dir)
            print("model_file from ", au, ": ", model_file)
            if (model_file == None):
                print(
                    "############################################################################################")
                print("####################################################################### None for ", au)
                print(
                    "############################################################################################")
            else:
                if FLAGS.test_iter > 0:
                    files = os.listdir(model_file[:model_file.index('model')])
                    if 'model' + str(FLAGS.test_iter) + '.index' in files:
                        model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
                        print("model_file by test_iter > 0: ", model_file)
                    else:
                        print(" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", files)
                print("Restoring model weights from " + model_file)

                saver.restore(sess, model_file)
                w = sess.run('model/w1:0')
                b = sess.run('model/b1:0')
                print("updated weights from ckpt: ", b)
                print('----------------------------------------------------------')
                if w_arr is None:
                    w_arr = w
                    b_arr = b
                else:
                    w_arr = np.hstack((w_arr, w))
                    b_arr = np.vstack((b_arr, b))

        return w_arr, b_arr

    def _load_weight_s(sbjt_start_idx):
        batch_size = 10
        # 모든 au 를 이용하여 한 모델을 만든경우 그 한 모델만 로드하면됨.
        if FLAGS.model.startswith('s1'):
            three_layers = feature_layer(batch_size, TOTAL_NUM_AU)
            three_layers.loadWeight(FLAGS.vae_model_to_test, FLAGS.au_idx, num_au_for_rm=TOTAL_NUM_AU)
        # 각 au별로 다른 모델인 경우 au별 weight을 쌓아줘야함
        else:
            three_layers = feature_layer(batch_size, 1)
            all_au = ['au1', 'au2', 'au4', 'au6', 'au9', 'au12', 'au25', 'au26']
            if FLAGS.au_idx < TOTAL_NUM_AU: all_au = [all_au[FLAGS.au_idx]]
            w_arr = None
            b_arr = None
            for au in all_au:
                if FLAGS.all_sub_model:  # s2, s3
                    three_layers.loadWeight(FLAGS.vae_model_to_test + '/' + FLAGS.model + '_' + au + '_kshot' + str(
                        FLAGS.update_batch_size) + '_iter100_subject' + str(sbjt_start_idx), au)
                else:  # only s4
                    three_layers.loadWeight(FLAGS.vae_model_to_test + '/s4_' + au + '_kshot' + str(
                        FLAGS.update_batch_size) + '_iter50_subject' + str(sbjt_start_idx), au)
                w = three_layers.model_intensity.layers[-1].get_weights()[0]
                b = three_layers.model_intensity.layers[-1].get_weights()[1]
                print('----------------------------------------------------------')
                if w_arr is None:
                    w_arr = w
                    b_arr = b
                else:
                    w_arr = np.hstack((w_arr, w))
                    b_arr = np.vstack((b_arr, b))

        return w_arr, b_arr



    def _load_weight_m0(trained_model_dir):
        model_file = None
        print('--------- model file dir: ', FLAGS.logdir + '/all_aus/' + trained_model_dir)
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/all_aus/' + trained_model_dir)
        print(">>>> model_file from all_aus: ", model_file)
        if (model_file == None):
            print("####################################################################### None for all_aus")
        else:
            if FLAGS.test_iter > 0:
                files = os.listdir(model_file[:model_file.index('model')])
                if 'model' + str(FLAGS.test_iter) + '.index' in files:
                    model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
                    print(">>>> model_file2: ", model_file)
                else:
                    print(" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", files)
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)
            w = sess.run('model/w1:0')
            b = sess.run('model/b1:0')
            print("updated weights from ckpt: ", b)
            print('----------------------------------------------------------')
        return w, b


    print("<<<<<<<<<<<< CONCATENATE >>>>>>>>>>>>>>")
    save_path = "./logs/result/"
    y_hat = []
    y_lab = []
    if FLAGS.all_sub_model:  # 모델이 모든 subjects를 이용해 train된 경우
        print('---------------- all sub model ----------------')

        ### test per each subject and concatenate
        for i in range(FLAGS.sbjt_start_idx, FLAGS.num_test_tasks):
            if FLAGS.model.startswith('s'):
                w_arr, b_arr = _load_weight_s(i)
            else:
                ### get path to load weight for 'm' models
                trained_model_dir = '/cls_' + str(FLAGS.num_classes) + '.mbs_' + str(
                    FLAGS.meta_batch_size) + '.ubs_' + str(
                    FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(
                    FLAGS.train_update_lr) + '.metalr' + str(FLAGS.meta_lr)
                w_arr, b_arr = _load_weight_m(trained_model_dir)  # weight load를 한번만 실행해도됨. subject별로 모델이 다르지 않기 때문
            result = test_each_subject(w_arr, b_arr, i)
            y_hat.append(result[0])
            y_lab.append(result[1])
            print("y_hat shape:", result[0].shape)
            print("y_lab shape:", result[1].shape)
            print(">> y_hat_all shape:", np.vstack(y_hat).shape)
            print(">> y_lab_all shape:", np.vstack(y_lab).shape)
        print_summary(np.vstack(y_hat), np.vstack(y_lab), log_dir=save_path + "/" + "test.txt")
    else:  # 모델이 각 subject 별로 train된 경우: vae와 MAML의 train_test두 경우에만 존재 가능 + local weight test의 경우
        for subj_idx in range(FLAGS.sbjt_start_idx, FLAGS.num_test_tasks):
            if FLAGS.model.startswith('s'):
                w_arr, b_arr = _load_weight_s(subj_idx)
            else:
                trained_model_dir = '/sbjt' + str(subj_idx) + '.ubs_' + str(
                    FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(
                    FLAGS.train_update_lr) + '.metalr' + str(FLAGS.meta_lr)
                w_arr, b_arr = _load_weight_m(trained_model_dir)
            result = test_each_subject(w_arr, b_arr, subj_idx)
            y_hat.append(result[0])
            y_lab.append(result[1])
            print("y_hat shape:", result[0].shape)
            print("y_lab shape:", result[1].shape)
            print(">> y_hat_all shape:", np.vstack(y_hat).shape)
            print(">> y_lab_all shape:", np.vstack(y_lab).shape)
        print_summary(np.vstack(y_hat), np.vstack(y_lab),
                      log_dir=save_path + "/test.txt")

    end_time = datetime.now()
    elapse = end_time - start_time
    print("=======================================================")
    print(">>>>>> elapse time: " + str(elapse))
    print("=======================================================")


if __name__ == "__main__":
    main()
