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


import pickle

start_time = datetime.now()
FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'disfa', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 2, 'number of classes used in classification (e.g. 5-way classification).')

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


flags.DEFINE_integer('sbjt_start_idx', 0, 'starting subject index')

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
flags.DEFINE_bool('all_sub_model', True, 'model is trained with all train/test tasks')
flags.DEFINE_string('model', "", 'model name')
flags.DEFINE_string('base_vae_model', "", 'base vae model to continue to train')
flags.DEFINE_string('norm', "batch_norm", '')
flags.DEFINE_integer('leave_one_out', -1, '')


def train(model, saver, sess, trained_model_dir, metatrain_input_tensors, resume_itr=0):
    print("===============> Final in weight: ", sess.run('model/w1:0').shape, sess.run('model/b1:0').shape)
    # print("===============> Final in weight: ", sess.run('model/w2:0').shape, sess.run('model/b2:0').shape)
    # print("===============> Final in weight: ", sess.run('model/w3:0').shape, sess.run('model/b3:0').shape)
    SUMMARY_INTERVAL = 500

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + trained_model_dir, sess.graph)

    feed_dict = {model.inputa: metatrain_input_tensors['inputa'].eval(),
                 model.inputb: metatrain_input_tensors['inputb'].eval(),
                 model.labela: metatrain_input_tensors['labela'].eval(),
                 model.labelb: metatrain_input_tensors['labelb'].eval(), model.meta_lr: FLAGS.meta_lr}

    print('Done initializing, starting training.')


    for itr in range(resume_itr + 1, FLAGS.metatrain_iterations + 1):
        if itr <= 1000:
            SAVE_INTERVAL = 100
        elif itr <= 5000:
            SAVE_INTERVAL = 1000
        else:
            SAVE_INTERVAL = 5000

        input_tensors = [model.metatrain_op]

        # when train the model again with the test set, local weight needs to be saved at the last iteration.

        if (itr % SUMMARY_INTERVAL == 0) or (itr == 1) or (itr == FLAGS.metatrain_iterations):
            input_tensors.extend([model.fast_weights])


        if (itr % SUMMARY_INTERVAL == 0) or (itr == 1) or (itr == FLAGS.metatrain_iterations):
            input_tensors.extend([model.result1, model.result2])

        result = sess.run(input_tensors, feed_dict)


        if itr % SUMMARY_INTERVAL == 0:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>summary")
            def summary(maml_result, set):
                print(set)
                print_str = 'Iteration ' + str(itr)
                print(print_str)
                y_hata = np.vstack(np.array(maml_result[-2][0]))  # length = num_of_task * N * K
                y_laba = np.vstack(np.array(maml_result[-2][1]))

                save_path = "./logs/result/train/" + trained_model_dir + "/" + str(FLAGS.au_idx) + "/"
                if FLAGS.keep_train_dir:
                    retrained_model_dir = 'sbjt' + str(FLAGS.sbjt_start_idx) + '.ubs_' + str(
                        FLAGS.train_update_batch_size) + '.numstep' + str(
                        FLAGS.num_updates) + '.updatelr' + str(
                        FLAGS.train_update_lr) + '.metalr' + str(FLAGS.meta_lr)
                    save_path += retrained_model_dir
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                print_summary(y_hata, y_laba, log_dir=save_path + "/outa_" + set + "_" + str(itr) + ".txt")
                print("------------------------------------------------------------------------------------")
                recent_y_hatb = np.array(maml_result[-1][0][
                                             FLAGS.num_updates - 1])
                y_hatb = np.vstack(recent_y_hatb)
                recent_y_labb = np.array(maml_result[-1][1][FLAGS.num_updates - 1])
                y_labb = np.vstack(recent_y_labb)
                print_summary(y_hatb, y_labb, log_dir=save_path + "/outb_" + set + "_" + str(itr) + ".txt")
                print("================================================================================")
            summary(result, "TR")

        if (itr % SAVE_INTERVAL == 0) or (itr == FLAGS.metatrain_iterations):
            print("======== w1 norm:", np.linalg.norm(sess.run('model/w1:0')))
            # print("======== w2 norm:", np.linalg.norm(sess.run('model/w2:0')))
            # print("======== w3 norm:", np.linalg.norm(sess.run('model/w3:0')))
            print('------------------------------ iter:', itr)
            # print("======== last weight :", w[-1])
            # print("======== b :", sess.run('model/b1:0'))
            if FLAGS.keep_train_dir:
                save_path = FLAGS.logdir + '/' + trained_model_dir
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                saver.save(sess, save_path + '/model' + str(itr))

                out = open(FLAGS.logdir + '/' + trained_model_dir + "/soft_weights" + str(itr) + ".pkl", 'wb')
                weights_to_save = {}
                for i in range(1):
                    weights_to_save['w' + str(i + 1)] = sess.run('model/w' + str(i + 1)+':0')
                    weights_to_save['b' + str(i + 1)] = sess.run('model/b' + str(i + 1)+':0')
                pickle.dump(weights_to_save, out, protocol=2)
                out.close()
                # save local weight at the last iteration
                # if itr == FLAGS.metatrain_iterations:
                #     print(">>>>>>>>>>>>>> local save !! : ", itr)
                #     local_w = result[1][0]
                #     local_b = result[1][1]
                #     print("================================================================================")
                #     print('>>>>>> Global bias: ', sess.run('model/b1:0'))
                #     local_model_dir = save_path + '/local'
                #     for i in range(FLAGS.meta_batch_size):
                #         model.weights['w1'].load(local_w[i], sess)
                #         model.weights['b1'].load(local_b[i], sess)
                #         print('>>>>>> Local bias for subject: ', i, sess.run('model/b1:0'))
                #         print("-----------------------------------------------------------------")
                #         if not os.path.exists(local_model_dir):
                #             os.makedirs(local_model_dir)
                #         saver.save(sess, local_model_dir + '/subject' + str(i))

            else:
                out = open(FLAGS.logdir + '/' + trained_model_dir + "/soft_weights" + str(itr) + ".pkl", 'wb')
                weights_to_save = {}
                for i in range(3):
                    weights_to_save['w' + str(i + 1)] = sess.run('model/w' + str(i + 1)+':0')
                    weights_to_save['b' + str(i + 1)] = sess.run('model/b' + str(i + 1)+':0')
                pickle.dump(weights_to_save, out, protocol=2)
                out.close()
                saver.save(sess, FLAGS.logdir + '/' + trained_model_dir + '/model' + str(itr))





def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

    data_generator = DataGenerator()

    dim_output = data_generator.num_classes
    dim_input = data_generator.dim_input

    inputa, inputb, labela, labelb = data_generator.make_data_tensor()
    metatrain_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    # pred_weights = data_generator.pred_weights
    model = MAML(dim_input, dim_output)
    model.construct_model(input_tensors=metatrain_input_tensors, prefix='metatrain_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=20)

    sess = tf.InteractiveSession()


    if not FLAGS.train:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

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

    if FLAGS.resume:
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
            b = sess.run('model/b1:0').tolist()
            print("updated weights from ckpt: ", np.array(b))
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1 + 5:])

    elif FLAGS.keep_train_dir:  # when the model needs to be initialized from another model.
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

    train(model, saver, sess, trained_model_dir, metatrain_input_tensors, resume_itr)

    end_time = datetime.now()
    elapse = end_time - start_time
    print("================================================================================")
    print(">>>>>> elapse time: " + str(elapse))
    print("================================================================================")


if __name__ == "__main__":
    main()
