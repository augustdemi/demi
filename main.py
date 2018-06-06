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
import csv
import numpy as np
import pickle
import random
import tensorflow as tf

from EmoEstimator.utils.evaluate import print_summary
from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags
from datetime import datetime
start_time = datetime.now()
FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'disfa', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 2, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 14, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
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
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('subject_idx', -1, 'subject index to test')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot
flags.DEFINE_bool('init_weight', True, 'Initialize weights from the base model')

def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    TEST_PRINT_INTERVAL = SUMMARY_INTERVAL * 5



    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')


    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        feed_dict = {}
        if itr < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]

        # SUMMARY_INTERVAL 혹은 PRINT_INTERVAL 마다 accuracy 계산해둠
        if (itr % SUMMARY_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])
            input_tensors.extend([model.result1, model.result2])
        result = sess.run(input_tensors, feed_dict)

        # SUMMARY_INTERVAL 마다 accuracy 쌓아둠
        if itr % SUMMARY_INTERVAL == 0:
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            if itr!=0:
                if itr < FLAGS.pretrain_iterations:
                    print_str = 'Pretrain Iteration ' + str(itr)
                else:
                    print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
                print(print_str)
                y_hata = np.vstack(np.array(result[-2][0])) #length = num_of_task * N * K
                y_laba = np.vstack(np.array(result[-2][1]))
                print_summary(y_hata, y_laba, log_dir="./logs/result/train/outa_" + str(itr) + ".txt")
                print("------------------------------------------------------------------------------------")
                recent_y_hatb = np.array(result[-1][0][FLAGS.num_updates-1])
                y_hatb = np.vstack(recent_y_hatb)
                recent_y_labb = np.array(result[-1][1][FLAGS.num_updates-1])
                y_labb = np.vstack(recent_y_labb)
                print_summary(y_hatb, y_labb, log_dir="./logs/result/train/outb_" + str(itr) + ".txt")
                print("====================================================================================")


        # SAVE_INTERVAL 마다 weight값 파일로 떨굼
        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))


    saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

# calculated for omniglot
NUM_TEST_POINTS = 1

def test(model, saver, sess, exp_string, data_generator):

    np.random.seed(1)
    random.seed(1)

    result_arr = []


    for _ in range(NUM_TEST_POINTS):
        feed_dict = {model.meta_lr: 0.0} # do not optimize in test
        # acc = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
        input_tensor = [model.metaval_result1, model.metaval_result2]
        result = sess.run(input_tensor, feed_dict)
        result_arr.append(result)
    y_hata = np.array(result[0][0])[0]
    y_laba = np.array(result[0][1])[0]
    print_summary(y_hata, y_laba, log_dir="./logs/result/test/outa_" + str(FLAGS.subject_idx) + ".txt")
    print("------------------------------------------------------------------------------------")
    y_hatb = np.mean(result[1][0], 0)[0]
    y_labb = np.mean(result[1][1], 0)[0]
    print_summary(y_hatb, y_labb, log_dir="./logs/result/test/outb_" + str(FLAGS.subject_idx) + ".txt")
    print("====================================================================================")


def main():

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

    data_generator = DataGenerator(FLAGS.update_batch_size * 2, FLAGS.meta_batch_size)

    dim_output = data_generator.num_classes
    if FLAGS.baseline == 'oracle':
        assert FLAGS.datasource == 'sinusoid'
        dim_input = 3
        FLAGS.pretrain_iterations += FLAGS.metatrain_iterations
        FLAGS.metatrain_iterations = 0
    else:
        dim_input = data_generator.dim_input

    num_classes = data_generator.num_classes

    if FLAGS.train:  # only construct training model if needed
        random.seed(5)
        image_tensor, label_tensor = data_generator.make_data_tensor()
        inputa = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1]) #(모든 task수, NK, 모든 dim) = (meta_batch_size, NK, 2000)
        #여기서 NK는 N개씩 K번 쌓은것. N개씩 쌓을때 0~N-1의 라벨을 하나씩 담되 랜덤 순서로 담음.
        inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])  #(모든 task수, NK, 모든 dim) = (meta_batch_size, NK, 2000)
        labela = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])  #(모든 task수, NK, 모든 label) = (meta_batch_size, NK, N)
        labelb = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1]) #(모든 task수, NK, 모든 label) = (meta_batch_size, NK, N)
        input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
    else:
        random.seed(6)
        image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
        inputa = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
        inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
        labela = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
        labelb = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
        metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    pred_weights = data_generator.pred_weights
    model = MAML(dim_input, dim_output)
    if FLAGS.train:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    else:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    sess = tf.InteractiveSession()


    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr) + '.initweight' + str(FLAGS.init_weight)


    if FLAGS.stop_grad:
        exp_string += 'stopgrad'
    if FLAGS.baseline:
        exp_string += FLAGS.baseline

    else:
        print('Norm setting not recognized.')





    resume_itr = 0


    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    # print(sess.run(metaval_input_tensors))
    # print(sess.run(input_tensors))
    print("========================================================================================")
    print('initial weights: ', sess.run(model.weights['w1']), sess.run('model/b1:0'))
    print('weights from vae : ', pred_weights)
    if FLAGS.init_weight:
        model.weights['w1'].load(pred_weights[0], sess)
        model.weights['b1'].load(pred_weights[1], sess)
    print('updated weights from vae: ', sess.run(model.weights['w1']), sess.run('model/b1:0'))
    print("========================================================================================")


    if FLAGS.resume or not FLAGS.train:
        model_file = None
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        print(">>>> model_file1: ", model_file)
        # model_file = tf.train.latest_checkpoint('../model.h5')
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
            print(">>>> model_file2: ", model_file)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)
            print("updated weights from ckpt: ", sess.run('model/w1:0'), sess.run('model/b1:0'))
            print('resume_itr: ', resume_itr)
            print("=====================================================================================")

    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        test(model, saver, sess, exp_string, data_generator)
    end_time = datetime.now()
    elapse = end_time - start_time
    print("=======================================================")
    print(">>>>>> elapse time: " + str(elapse))
    print("=======================================================")
if __name__ == "__main__":
    main()
