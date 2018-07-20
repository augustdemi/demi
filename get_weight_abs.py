"""
--train=False --test_set=True --subject_idx=14 --num_classes=2 --datasource=disfa --metatrain_iterations=10 --meta_batch_size=14 --update_batch_size=1 --update_lr=0.4 --num_updates=5 --logdir=logs/disfa/
Usage Instructions:
    10-shot sinusoid:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10

    10-shot sinusoid baselines:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10 --baseline=oracle
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10

    5-way, 1-shot omniglot:

    20-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=logs/omniglot20way/

    5-way 1-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet1shot/ --num_filters=32 --max_pool=True

    5-way 5-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=5 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet5shot/ --num_filters=32 --max_pool=True

    To run evaluation, use the '--train=False' flag and the '--test_set=True' flag to use the test set.

    For omniglot and miniimagenet training, acquire the dataset online, put it in the correspoding data directory, and see the python script instructions in that directory to preprocess the data.
"""
import numpy as np
import tensorflow as tf

from EmoEstimator.utils.evaluate import print_summary
from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags
from datetime import datetime
import os

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
flags.DEFINE_string('datadir', '/home/ml1323/project/robert_data/DISFA/kshot/0', 'directory for data.')
flags.DEFINE_string('valdir', '/home/ml1323/project/robert_data/DISFA/kshot/1', 'directory for val.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_integer('num_test_pts', 1, 'number of iteration to increase the test points')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('subject_idx', -1, 'subject index to test')
flags.DEFINE_integer('train_update_batch_size', -1,
                     'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1,
                   'value of inner gradient step step during training. (use if you want to test with a different value)')  # 0.1 for omniglot

flags.DEFINE_bool('init_weight', False, 'Initialize weights from the base model')
flags.DEFINE_bool('train_test', False, 're-train model with the test set')
flags.DEFINE_bool('train_test_inc', False, 're-train model increasingly')
flags.DEFINE_bool('test_test', False, 'test the test set with test-model')
flags.DEFINE_bool('test_train', False, 'test the test set with train-model')
# for train, train_test
flags.DEFINE_integer('train_start_idx', 0, 'start index of task for training')
# for test_test, test_train
flags.DEFINE_integer('test_start_idx', 14, 'start index of task for test')
flags.DEFINE_integer('test_num', 1, 'num of task for test')
flags.DEFINE_string('testset_dir', './data/1/', 'directory for test set')
flags.DEFINE_string('test_result_dir', 'robert', 'directory for test result log')
# for train_test, test_test
flags.DEFINE_string('keep_train_dir', None,
                    'directory to read already trained model when training the model again with test set')
flags.DEFINE_integer('local_subj', 0, 'local weight subject')
flags.DEFINE_string('vae_model', './model78.h5', 'vae model dir from robert code')



def main():
    temp = FLAGS.update_batch_size
    temp2 = FLAGS.meta_batch_size

    FLAGS.update_batch_size=1
    FLAGS.meta_batch_size=1
    data_generator = DataGenerator(FLAGS.update_batch_size * 2, FLAGS.meta_batch_size)

    dim_output = data_generator.num_classes
    dim_input = data_generator.dim_input


    if FLAGS.train:  # only construct training model if needed

        # image_tensor, label_tensor = data_generator.make_data_tensor()
        # inputa = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1]) #(모든 task수, NK, 모든 dim) = (meta_batch_size, NK, 2000)
        # #여기서 NK는 N개씩 K번 쌓은것. N개씩 쌓을때 0~N-1의 라벨을 하나씩 담되 랜덤 순서로 담음.
        # inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])  #(모든 task수, NK, 모든 dim) = (meta_batch_size, NK, 2000)
        # labela = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])  #(모든 task수, NK, 모든 label) = (meta_batch_size, NK, N)
        # labelb = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1]) #(모든 task수, NK, 모든 label) = (meta_batch_size, NK, N)
        inputa, inputb, labela, labelb = data_generator.make_data_tensor()
        metatrain_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    inputa, inputb, labela, labelb = data_generator.make_data_tensor(train=False)
    metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    pred_weights = data_generator.pred_weights
    model = MAML(dim_input, dim_output)
    if FLAGS.train:
        model.construct_model(input_tensors=metatrain_input_tensors, prefix='metatrain_')
    else:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=20)

    sess = tf.InteractiveSession()

    FLAGS.update_batch_size = temp
    FLAGS.meta_batch_size = temp2

    trained_model_dir = 'cls_' + str(FLAGS.num_classes) + '.mbs_' + str(FLAGS.meta_batch_size) + '.ubs_' + str(
        FLAGS.update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(
        FLAGS.update_lr) + '.metalr' + str(FLAGS.meta_lr) + '.initweight' + str(FLAGS.init_weight) + \
                        '/sbjt14:13.ubs_' + str(FLAGS.update_batch_size) +'.numstep5.updatelr0.005.metalr0.005'


    # if FLAGS.stop_grad:
    #     trained_model_dir += 'stopgrad'
    # if FLAGS.baseline:
    #     trained_model_dir += FLAGS.baseline
    # else:
    #     print('Norm setting not recognized.')


    resume_itr = 0

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()


    model_file = None
    model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + trained_model_dir)
    w = None
    b = None
    print(">>> kshot: ", FLAGS.update_batch_size)
    print(">>>> train_test model dir: ", model_file)
    model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + trained_model_dir)
    saver.restore(sess, model_file)
    w = sess.run('model/w1:0')
    print("global abs of w: ", np.linalg.norm(w))
    b = sess.run('model/b1:0')
    print("global abs of b: ", np.linalg.norm(b))
    model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + trained_model_dir + '/local')
    for i in range(13):
        model_file = model_file[:model_file.index('subject')] + 'subject' + str(i)
        print(">>>> model_file_local: ", model_file)
        saver.restore(sess, model_file)
        w = sess.run('model/w1:0')
        print("subject ", i, ", abs of w: ", np.linalg.norm(w))
        b = sess.run('model/b1:0')
        print("subject ", i, ", abs of b: ", np.linalg.norm(b))



if __name__ == "__main__":
    main()
