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
from data_generator2 import DataGenerator
from maml_new import MAML
from tensorflow.python.platform import flags
from datetime import datetime
import os

import pickle

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
flags.DEFINE_integer('subject_idx', -1, 'subject index to test')
flags.DEFINE_integer('train_update_batch_size', -1,
                     'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1,
                   'value of inner gradient step step during training. (use if you want to test with a different value)')  # 0.1 for omniglot

flags.DEFINE_bool('init_weight', True, 'Initialize weights from the base model')
flags.DEFINE_bool('train_train', False, 're-train model with the train')
flags.DEFINE_bool('train_test', False, 're-train model with the test set')

# for train, train_test
flags.DEFINE_integer('sbjt_start_idx', 0, 'starting subject index')

# for train_test, test_test
flags.DEFINE_string('keep_train_dir', None,
                    'directory to read already trained model when training the model again with test set')
flags.DEFINE_integer('local_subj', 0, 'local weight subject')
flags.DEFINE_integer('kshot_seed', 0, 'seed for k shot sampling')
flags.DEFINE_integer('weight_seed', 0, 'seed for initial weight')
flags.DEFINE_integer('num_au', 8, 'number of AUs used to make AE')
flags.DEFINE_integer('au_idx', 8, 'au index to use in the given AE')
flags.DEFINE_string('vae_model', './model_au_12.h5', 'vae model dir from robert code')
flags.DEFINE_string('gpu', "0,1,2,3", 'vae model dir from robert code')
flags.DEFINE_string('feature_path', "", 'path for feature vector')
flags.DEFINE_bool('temp_train', False, 'test the test set with train-model')
flags.DEFINE_bool('all_sub_model', True, 'model is trained with all train/test tasks')
flags.DEFINE_bool('meta_update', True, 'meta_update')
flags.DEFINE_string('model', "", 'model name')
flags.DEFINE_string('base_vae_model', "", 'base vae model to continue to train')
flags.DEFINE_string('opti', '', 'optimizer : adam or adadelta')
flags.DEFINE_integer('shuffle_batch', 1, '')
flags.DEFINE_float('lambda2', 0.5, '')
flags.DEFINE_string('adaptation', "", 'adaptation way: inner or outer')

def train(model, metatrain_input_tensors, saver, sess, trained_model_dir, resume_itr=0):
    print("===============> Final in weight: ", sess.run('model/w1:0').shape, sess.run('model/b1:0').shape)
    SUMMARY_INTERVAL = 10
    SAVE_INTERVAL = 5000

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + trained_model_dir, sess.graph)

    data_generator = DataGenerator()

    feed_dict = {model.inputa: metatrain_input_tensors['inputa'].eval(),
                 model.inputb: metatrain_input_tensors['inputb'].eval(),
                 model.labela: metatrain_input_tensors['labela'].eval(),
                 model.labelb: metatrain_input_tensors['labelb'].eval(), model.meta_lr: FLAGS.meta_lr}

    print('Done initializing, starting training.')

    for itr in range(resume_itr + 1, FLAGS.metatrain_iterations + 1):
        if itr % FLAGS.shuffle_batch == 0:
            print('=============================================================shuffle data, iteration:', itr)
            inputa, inputb, labela, labelb = data_generator.make_data_tensor(itr)
            feed_dict = {model.inputa: inputa.eval(),
                         model.inputb: inputb.eval(),
                         model.labela: labela.eval(),
                         model.labelb: labelb.eval(), model.meta_lr: FLAGS.meta_lr}

        if itr <= 1000:
            SAVE_INTERVAL = 100
        elif itr <= 5000:
            SAVE_INTERVAL = 1000
        else:
            SAVE_INTERVAL = 5000

        input_tensors = [model.train_op]

        if (itr % SUMMARY_INTERVAL == 0):
            input_tensors.extend([model.summ_op])

        input_tensors.extend([model.fast_weight_w])
        input_tensors.extend([model.fast_weight_b])
        result = sess.run(input_tensors, feed_dict)

        if (itr % SUMMARY_INTERVAL == 0):
            train_writer.add_summary(result[1], itr)

        if FLAGS.train_test:
            local_model_dir = FLAGS.keep_train_dir + '/adaptation.' + FLAGS.adaptation + '.kshot' + FLAGS.update_batch_size + '.update_lr' + str(
                FLAGS.update_lr) + '.metalr' + str(FLAGS.meta_lr) + '.lambda' + str(
                FLAGS.lambda2) + '.num_updates' + str(FLAGS.num_updates) + '.meta_iter' + str(
                FLAGS.metatrain_iterations) + '.opti' + FLAGS.opti
            if not os.path.exists(local_model_dir):
                os.makedirs(local_model_dir)
            print("================================================ iter:", itr)
            print('>>>>>>  subject : ', FLAGS.sbjt_start_idx)
            if FLAGS.adaptation.startswith('outer'):
                w = sess.run('model/w1:0')
                print()
                print("= weight norm:", np.linalg.norm(w))
                print("= last weight :", w[-1])
                print("= b :", sess.run('model/b1:0'))
                print("= w shape :", sess.run('model/w1:0').shape)

                out = open(local_model_dir + '/subject' + str(FLAGS.sbjt_start_idx) + ".pkl", 'wb')
                weights_to_save = {}
                weights_to_save.update({'w': sess.run('model/w1:0')})
                weights_to_save.update({'b': sess.run('model/b1:0')})
                pickle.dump(weights_to_save, out, protocol=2)
                out.close()
            elif FLAGS.adaptation.startswith('inner'):
                assert (FLAGS.metatrain_iterations == 1)
                # save local weight at the last iteration
                print(">>>>>>>>>>>>>> local save !! : ", itr)
                fast_w = np.array(result[-2])
                fast_b = np.array(result[-1])
                print("fast_w shape: ", fast_w.shape)
                print("fast_b shape: ", fast_b.shape)
                print("================================================================================")
                print('>>>>>> Global bias: ', sess.run('model/b1:0'))
                for i in range(FLAGS.meta_batch_size):
                    print('>>>>>>  subject : ', i)
                    out = open(local_model_dir + '/subject' + str(i) + ".pkl", 'wb')
                    weights_to_save = {}
                    weights_to_save.update({'w': fast_w[:, i]})
                    weights_to_save.update({'b': fast_b[:, i]})
                    pickle.dump(weights_to_save, out, protocol=2)
                    out.close()
            else:
                print(">>>>>>>>>>>>>> check adaptation method: inner or outer but given ", FLAGS.adaptation)



        elif (itr % SAVE_INTERVAL == 0) or (itr == FLAGS.metatrain_iterations):
            w = sess.run('model/w1:0')
            print("================================================ iter:", itr)
            print()
            print("= weight norm:", np.linalg.norm(w))
            print("= last weight :", w[-1])
            print("= b :", sess.run('model/b1:0'))
            print("= b :", sess.run('model/w1:0').shape)
            out = open(FLAGS.logdir + '/' + trained_model_dir + "/soft_weights" + str(itr) + ".pkl", 'wb')
            pickle.dump({'w': sess.run('model/w1:0'), 'b': sess.run('model/b1:0')}, out, protocol=2)
            out.close()
            saver.save(sess, FLAGS.logdir + '/' + trained_model_dir + '/model' + str(itr))


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    data_generator = DataGenerator()

    dim_output = data_generator.num_classes
    dim_input = data_generator.dim_input

    inputa, inputb, labela, labelb = data_generator.make_data_tensor(0)
    metatrain_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    # pred_weights = data_generator.pred_weights
    model = MAML(dim_input, dim_output)
    model.construct_model(input_tensors=metatrain_input_tensors, prefix='metatrain_')
    model.summ_op = tf.summary.merge_all()

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=20)

    sess = tf.InteractiveSession()

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    trained_model_dir = 'cls_' + str(FLAGS.num_classes) + '.mbs_' + str(FLAGS.meta_batch_size) + '.ubs_' + str(
        FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(
        FLAGS.train_update_lr) + '.metalr' + str(FLAGS.meta_lr)

    print(">>>>> trained_model_dir: ", FLAGS.logdir + '/' + trained_model_dir)

    resume_itr = 0

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    print("================================================================================")
    print('initial weights norm: ', np.linalg.norm(sess.run('model/w1:0')))
    print('initial last weights: ', sess.run('model/w1:0')[-1])
    print('initial bias: ', sess.run('model/b1:0'))
    print("================================================================================")

    ################## Train ##################

    # train_train or train_test
    if FLAGS.resume:  # 디폴트로 resume은 항상 true. 따라서 train중간부터 항상 시작 가능.
        model_file = None
        if FLAGS.model.startswith('m2'):
            trained_model_dir = 'sbjt' + str(FLAGS.sbjt_start_idx) + '.ubs_' + str(
                FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(
                FLAGS.train_update_lr) + '.metalr' + str(FLAGS.meta_lr)
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + trained_model_dir)
        print(">>>>> trained_model_dir: ", FLAGS.logdir + '/' + trained_model_dir)

        w = None
        b = None
        print(">>>> model_file1: ", model_file)

        if model_file:
            if FLAGS.test_iter > 0:
                files = os.listdir(model_file[:model_file.index('model')])
                if 'model' + str(FLAGS.test_iter) + '.index' in files:
                    model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
                    print(">>>> model_file2: ", model_file)
            print("1. Restoring model weights from " + model_file)
            saver.restore(sess, model_file)
            w = sess.run('model/w1:0').tolist()
            b = sess.run('model/b1:0').tolist()
            print("updated weights from ckpt: ", np.array(b))
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1 + 5:])
            print('resume_itr: ', resume_itr)

    elif FLAGS.train_test or FLAGS.train_train:  # train_test의 첫 시작인 경우 resume은 false이지만 trained maml로 부터 모델 로드는 해야함.
        resume_itr = 0
        print('resume_itr: ', resume_itr)
        model_file = tf.train.latest_checkpoint(FLAGS.keep_train_dir)
        print(">>>>> base_model_dir: ", FLAGS.keep_train_dir)

        if FLAGS.test_iter > 0:
            files = os.listdir(model_file[:model_file.index('model')])
            if 'model' + str(FLAGS.test_iter) + '.index' in files:
                model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
                print(">>>> model_file2: ", model_file)

        print("2. Restoring model weights from " + model_file)
        saver.restore(sess, model_file)
        print("updated weights from ckpt: ", sess.run('model/b1:0'))

    elif FLAGS.model.startswith('s4'):
        from feature_layers import feature_layer
        three_layers = feature_layer(10, 1)
        print('FLAGS.base_vae_model: ', FLAGS.base_vae_model)
        three_layers.model_intensity.load_weights(FLAGS.base_vae_model + '.h5')
        w = three_layers.model_intensity.layers[-1].get_weights()[0]
        b = three_layers.model_intensity.layers[-1].get_weights()[1]
        print('s2 b: ', b)
        print('s2 w: ', w)
        print('-----------------------------------------------------------------')
        with tf.variable_scope("model", reuse=True) as scope:
            scope.reuse_variables()
            b1 = tf.get_variable("b1", [1, 2]).assign(np.array(b))
            w1 = tf.get_variable("w1", [300, 1, 2]).assign(np.array(w))
            sess.run(b1)
            sess.run(w1)
        print("after: ", sess.run('model/b1:0'))
        print("after: ", sess.run('model/w1:0'))
    if not FLAGS.all_sub_model:
        trained_model_dir = 'sbjt' + str(FLAGS.sbjt_start_idx) + '.ubs_' + str(
            FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(
            FLAGS.train_update_lr) + '.metalr' + str(FLAGS.meta_lr)

    print("================================================================================")

    train(model, metatrain_input_tensors, saver, sess, trained_model_dir, resume_itr)

    end_time = datetime.now()
    elapse = end_time - start_time
    print("================================================================================")
    print(">>>>>> elapse time: " + str(elapse))
    print("================================================================================")


if __name__ == "__main__":
    main()
