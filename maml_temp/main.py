"""
--train=False --test_set=True --subject_idx=14
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

from maml_temp.data_generator import DataGenerator
from maml_temp.maml import MAML
from tensorflow.python.platform import flags

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
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('subject_idx', -1, 'subject index to test')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    PRINT_INTERVAL = 100
    TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5


    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []

    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        print(iter, " starts")
        feed_dict = {}
        if itr < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            if model.classification:
                input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])

        result = sess.run(input_tensors, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        # sinusoid is infinite data, so no need to test on meta-validation set.
        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
            input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates - 1],
                             model.summ_op]
            result = sess.run(input_tensors, feed_dict)

            print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

    saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

# calculated for omniglot
NUM_TEST_POINTS = 600

def test(model, saver, sess, exp_string, data_generator):

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []

    from EmoEstimator.utils.evaluate import print_summary
    for _ in range(1):
        feed_dict = {model.meta_lr: 0.0}
        # acc = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
        result = sess.run(model.metaval_result1, feed_dict)
        result2 = sess.run(model.metaval_result2, feed_dict)
        metaval_accuracies.append(result)
    print("------------------------------------------")
    y_hata = np.array(result[0])[0]
    y_laba = np.array(result[1])[0]
    print_summary(y_hata, y_laba)
    print("------------------------------------------")
    y_hatb = np.mean(result2[0], 0)[0]
    y_labb = np.mean(result2[1], 0)[0]
    print_summary(y_hatb, y_labb)
    print("------------------------------------------")

    # metaval_accuracies = np.array(metaval_accuracies)
    # print(len(metaval_accuracies))
    # means = np.mean(metaval_accuracies, 0)
    # stds = np.std(metaval_accuracies, 0)
    # ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)
    #
    #
    # print('Mean validation accuracy/loss, stddev, and confidence intervals')
    # print((means, stds, ci95))
    #
    # out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
    # out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.pkl'
    # with open(out_pkl, 'wb') as f:
    #     pickle.dump({'mses': metaval_accuracies}, f)
    # with open(out_filename, 'w') as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerow(['update'+str(i) for i in range(len(means))])
    #     writer.writerow(means)
    #     writer.writerow(stds)
    #     writer.writerow(ci95)

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
        inputa = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
        inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
        labela = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
        labelb = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
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

    exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)


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

if __name__ == "__main__":
    main()
