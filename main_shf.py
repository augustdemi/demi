import numpy as np
import tensorflow as tf
from datetime import datetime
import os
import pickle

from data_generator_shf import DataGenerator
from maml_shf import MAML
from tensorflow.python.platform import flags
from two_layer import feature_layer


start_time = datetime.now()
FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_integer('num_classes', 2, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
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
flags.DEFINE_integer('train_update_batch_size', -1,
                     'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1,
                   'value of inner gradient step step during training. (use if you want to test with a different value)')  # 0.1 for omniglot

# for train, train_test
flags.DEFINE_integer('sbjt_start_idx', 0, 'starting subject index')

# for train_test, test_test
flags.DEFINE_string('keep_train_dir', None,
                    'directory to read already trained model when training the model again with test set')
flags.DEFINE_integer('kshot_seed', 0, 'seed for k shot sampling')
flags.DEFINE_integer('weight_seed', 0, 'seed for initial weight')
flags.DEFINE_integer('num_au', 8, 'number of AUs used to make AE')
flags.DEFINE_integer('au_idx', 8, 'au index to use in the given AE')
flags.DEFINE_string('vae_model', './model_au_12.h5', 'vae model dir from robert code')
flags.DEFINE_string('gpu', "0,1,2,3", 'vae model dir from robert code')
flags.DEFINE_string('kshot_path', "", 'kshot csv path')
flags.DEFINE_bool('meta_update', True, 'meta_update')
flags.DEFINE_string('model', "", 'model name')
flags.DEFINE_string('base_vae_model', None, 'base vae model to continue to train')
flags.DEFINE_string('opti', '', 'optimizer : adam or adadelta')
flags.DEFINE_integer('shuffle_batch', -1, '')
flags.DEFINE_float('lambda2', 0.5, '')
flags.DEFINE_bool('adaptation', False, 'adaptation or not')
flags.DEFINE_string('labeldir', "/home/ml1323/project/robert_data/DISFA/label/", 'label_dir')
flags.DEFINE_string('check_sample', None, 'check frame idx of samples')
flags.DEFINE_integer('test_split_seed', -1, 'random seed for test set split')
flags.DEFINE_integer('feat_dim', 300, 'input feature dimension')
flags.DEFINE_bool('evaluate', False, 'evaluate or not')
flags.DEFINE_string('subject_index', '', 'subject indices to select')
flags.DEFINE_bool('init', False, 'initialize weight from vae_model')

def train(model, data_generator, saver, sess, trained_model_dir, resume_itr=0):
    print("===============> Final in weight: ", sess.run('model/w1:0').shape, sess.run('model/b1:0').shape)
    SUMMARY_INTERVAL = 10
    SAVE_INTERVAL = 5000

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + trained_model_dir, sess.graph)

    feed_dict = {}

    print('Done initializing, starting training.')
    aus = ['au1', 'au2', 'au4', 'au6', 'au9', 'au12', 'au25', 'au26']

    for itr in range(resume_itr + 1, FLAGS.metatrain_iterations + 1):
        if FLAGS.shuffle_batch > 0 and itr % FLAGS.shuffle_batch == 0:
            print('=============================================================shuffle data, iteration:', itr)
            inputa, inputb, labela, labelb, _ = data_generator.shuffle_data(itr, FLAGS.update_batch_size, aus)
            feed_dict = {model.inputa: inputa,
                         model.inputb: inputb,
                         model.labela: labela,
                         model.labelb: labelb}

        if itr <= 1000:
            SAVE_INTERVAL = 100
        elif itr <= 5000:
            SAVE_INTERVAL = 1000
        else:
            SAVE_INTERVAL = 2000

        input_tensors = [model.train_op]

        if (itr % SUMMARY_INTERVAL == 0):
            input_tensors.extend([model.summ_op])

        input_tensors.extend([model.fast_weight_w])
        input_tensors.extend([model.fast_weight_b])
        result = sess.run(input_tensors, feed_dict)

        if (itr % SUMMARY_INTERVAL == 0):
            train_writer.add_summary(result[1], itr)


        if (itr % SAVE_INTERVAL == 0) or (itr == FLAGS.metatrain_iterations):
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

        if itr == FLAGS.metatrain_iterations:
            with open(FLAGS.logdir + '/' + trained_model_dir + "/per_sub_weight.pkl", 'wb') as out:
                pickle.dump({'w': result[-2], 'b': result[-1]}, out, protocol=2)
                out.close()


def test(model, sess, trained_model_dir, data_generator, all_used_frame_set):
    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + trained_model_dir, sess.graph)

    feed_dict = {}
    print('Done initializing, starting training.')
    w = sess.run('model/w1:0')
    b = sess.run('model/b1:0')
    if FLAGS.base_vae_model:
        adapted_model_dir = FLAGS.keep_train_dir + '/adaptation_base/update_lr' + str(
            FLAGS.update_lr) + '.metalr' + str(FLAGS.meta_lr) + '.lambda' + str(
            FLAGS.lambda2) + '.num_updates' + str(FLAGS.num_updates) + '.meta_iter' + str(
            FLAGS.metatrain_iterations) + '/splitseed' + str(
            FLAGS.test_split_seed) + '.' + str(FLAGS.update_batch_size) + 'shot.kseed' + str(FLAGS.kshot_seed)
    elif FLAGS.keep_train_dir:
        adapted_model_dir = FLAGS.keep_train_dir + '/adaptation/update_lr' + str(
            FLAGS.update_lr) + '.metalr' + str(FLAGS.meta_lr) + '.lambda' + str(
            FLAGS.lambda2) + '.num_updates' + str(FLAGS.num_updates) + '.meta_iter' + str(
            FLAGS.metatrain_iterations) + '/splitseed' + str(
            FLAGS.test_split_seed) + '.' + str(FLAGS.update_batch_size) + 'shot.kseed' + str(FLAGS.kshot_seed)
    else:
        adapted_model_dir = './validation/adaptation/update_lr' + str(
            FLAGS.update_lr) + '.metalr' + str(FLAGS.meta_lr) + '.lambda' + str(
            FLAGS.lambda2) + '.num_updates' + str(FLAGS.num_updates) + '.meta_iter' + str(
            FLAGS.metatrain_iterations) + '/splitseed' + str(
            FLAGS.test_split_seed) + '.' + str(FLAGS.update_batch_size) + 'shot.kseed' + str(FLAGS.kshot_seed) + '.wseed' + str(FLAGS.weight_seed)
    if FLAGS.test_iter > 0:
        adapted_model_dir = adapted_model_dir + '/test_iter' + str(FLAGS.test_iter)

    if not os.path.exists(adapted_model_dir):
        os.makedirs(adapted_model_dir)

    for itr in range(1, FLAGS.metatrain_iterations + 1):
        input_tensors = [model.train_op]
        input_tensors.extend([model.fast_weight_w])
        input_tensors.extend([model.fast_weight_b])
        result = sess.run(input_tensors, feed_dict)

        if itr == FLAGS.metatrain_iterations:
            print("================================================ iter {}, subject {}".format(itr,
                                                                                                FLAGS.sbjt_start_idx))
            w = sess.run('model/w1:0')
            b = sess.run('model/b1:0')
            print('--- adapted bias: ', b)
            out = open(adapted_model_dir + '/subject' + str(FLAGS.sbjt_start_idx) + ".pkl", 'wb')
            pickle.dump({'w': w, 'b': b}, out, protocol=2)
            out.close()
    if FLAGS.evaluate:

        soft_layer = feature_layer(10, FLAGS.num_au)
        soft_layer.loadWeight(FLAGS.vae_model, w=w, b=b)
        print('--- loaded bias to be evaluated: ', b)

        subjects = os.listdir(FLAGS.datadir)
        subjects.sort()


        if FLAGS.train:
            adapted_model_dir += '/val'
            f = pickle.load(open('./validation/maml/fold1/m1_ce_0.01co_shuffle1_adadelta_each/cls_2.mbs_18.ubs_10.numstep20.updatelr0.01.metalr0.01.initFalse/per_sub_weight.pkl', 'rb'), encoding='latin1')
            w = np.array(f['w'])
            b = np.array(f['b'])

            w_all = []
            b_all = []
            for i in range(18):
                w_per_subject = w[0, i]
                b_per_subject = b[0, i]
                for j in range(1, 8):
                    w_per_subject = np.append(w_per_subject, w[j, i], axis=1)
                    b_per_subject = np.append(b_per_subject, b[j, i], axis=0)
                w_all.append(w_per_subject)
                b_all.append(b_per_subject)


            print('total labels: ', len(data_generator.labels))
            print('total subjects: ', subjects)
            for i in range(len(subjects)):
                eval_vec = []
                eval_frame = []
                # with tf.variable_scope("model", reuse=True) as scope:
                #     scope.reuse_variables()
                #     b1 = tf.get_variable("b1", [FLAGS.num_au, 2]).assign(b_all[i])
                #     w1 = tf.get_variable("w1", [300, FLAGS.num_au, 2]).assign(w_all[i])
                #     sess.run(b1)
                #     sess.run(w1)
                # print("uploaded bias from per sub weight for subject {} : {}".format(i, sess.run('model/b1:0')))


                print('-- evaluate vec: ', subjects[i])
                with open(os.path.join(FLAGS.datadir, subjects[i]), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.split(',')
                        frame_idx = int(line[1].split('frame')[1])
                        if frame_idx < 4845:
                            feat_vec = [float(elt) for elt in line[2:]]
                            eval_vec.append(feat_vec)
                            eval_frame.append(frame_idx)

                y_lab = data_generator.labels[i][eval_frame]
                print('----------------eval_frame ---------------------')
                print(len(eval_frame))
                # print(eval_frame)
                print('----------------y_lab ---------------------')
                print(len(y_lab))
                # print(y_lab)
                y_lab = np.array([np.eye(2)[label] for label in y_lab])
                y_hat = soft_layer.model_intensity.predict(eval_vec)
                print('y_lab shape: ', y_lab.shape)
                print('y_hat shape: ', y_hat.shape)
                out = open(adapted_model_dir + '/predicted_subject' + str(i) + ".pkl", 'wb')
                pickle.dump({'y_lab': y_lab, 'y_hat': y_hat, 'all_used_frame_set': all_used_frame_set}, out, protocol=2)
                out.close()
        else:
            print('-- evaluate vec: ', subjects[FLAGS.sbjt_start_idx])
            eval_vec = []
            eval_frame = []
            with open(os.path.join(FLAGS.datadir, subjects[FLAGS.sbjt_start_idx]), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split(',')
                    frame_idx = int(line[1].split('frame')[1])
                    if frame_idx in data_generator.test_b_frame and frame_idx < 4845:
                        feat_vec = [float(elt) for elt in line[2:]]
                        eval_vec.append(feat_vec)
                        eval_frame.append(frame_idx)
            y_lab = data_generator.labels[0][eval_frame]
            y_lab = np.array([np.eye(2)[label] for label in y_lab])
            y_hat = soft_layer.model_intensity.predict(eval_vec)
            print('y_lab shape: ', y_lab.shape)
            print('y_hat shape: ', y_hat.shape)
            out = open(adapted_model_dir + '/predicted_subject' + str(FLAGS.sbjt_start_idx) + ".pkl", 'wb')
            pickle.dump({'y_lab': y_lab, 'y_hat': y_hat, 'all_used_frame_set': all_used_frame_set}, out, protocol=2)
            out.close()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4

    data_generator = DataGenerator()

    aus = ['au1', 'au2', 'au4', 'au6', 'au9', 'au12', 'au25', 'au26']
    if FLAGS.adaptation:
        inputa, inputb, labela, labelb, all_used_frame_set = data_generator.sample_test_data_use_all(FLAGS.kshot_seed,
                                                                                                     FLAGS.update_batch_size,
                                                                                                     aus)
    else:
        inputa, inputb, labela, labelb, all_used_frame_set = data_generator.shuffle_data(FLAGS.kshot_seed,
                                                                                         FLAGS.update_batch_size, aus)
        # val_input, val_label =


    # inputa = (aus*subjects, 2K, latent_dim)
    # labela = (aus*subjects, 2K, au)
    metatrain_input_tensors = {'inputa': tf.convert_to_tensor(inputa), 'inputb': tf.convert_to_tensor(inputb),
                               'labela': tf.convert_to_tensor(labela), 'labelb': tf.convert_to_tensor(labelb)}

    dim_input = FLAGS.feat_dim  # img size
    model = MAML(dim_input, FLAGS.num_classes)
    model.construct_model(input_tensors=metatrain_input_tensors)
    model.summ_op = tf.summary.merge_all()

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=20)

    sess = tf.InteractiveSession()

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    trained_model_dir = 'cls_' + str(FLAGS.num_classes) + '.mbs_' + str(FLAGS.meta_batch_size) + '.ubs_' + str(
        FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(
        FLAGS.train_update_lr) + '.metalr' + str(FLAGS.meta_lr) + '.init' + str(FLAGS.init)

    resume_itr = 0
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    ################## Train ##################
    if FLAGS.init:
        print('FLAGS.vae_model: ', FLAGS.vae_model)
        soft_layer = feature_layer(10, FLAGS.num_au)
        soft_layer.loadWeight(FLAGS.vae_model)
        w = soft_layer.model_intensity.layers[-1].get_weights()[0]
        b = soft_layer.model_intensity.layers[-1].get_weights()[1]
        print('bias from base_vae_model: ', b)
        print('-----------------------------------------------------------------')
        with tf.variable_scope("model", reuse=True) as scope:
            scope.reuse_variables()
            b1 = tf.get_variable("b1", [FLAGS.num_au, 2]).assign(np.array(b))
            w1 = tf.get_variable("w1", [300, FLAGS.num_au, 2]).assign(np.array(w))
            sess.run(b1)
            sess.run(w1)
        print("uploaded bias from vae_model: ", sess.run('model/b1:0'))


    if FLAGS.resume:
        model_file = None
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + trained_model_dir)
        print(">>>>> trained_model_dir: ", FLAGS.logdir + '/' + trained_model_dir)
        print(">>>> model_file1: ", model_file)
        if model_file:
            if FLAGS.test_iter > 0:
                files = os.listdir(model_file[:model_file.index('model')])
                if 'model' + str(FLAGS.test_iter) + '.index' in files:
                    model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
            print("1. Restoring model weights from " + model_file)
            saver.restore(sess, model_file)
            b = sess.run('model/b1:0').tolist()
            print("updated bias from ckpt: ", np.array(b))
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1 + 5:])
            print('resume_itr: ', resume_itr)

    elif FLAGS.adaptation:  # adaptation 첫 시작인 경우 resume은 false이지만 trained maml로 부터 모델 로드는 해야함.
        if FLAGS.base_vae_model:
            print('FLAGS.base_vae_model: ', FLAGS.base_vae_model)
            soft_layer = feature_layer(10, FLAGS.num_au)
            if 'sub' in FLAGS.base_vae_model:
                soft_layer.model_intensity.load_weights(FLAGS.base_vae_model + '.h5')
            else:
                soft_layer.loadWeight(FLAGS.base_vae_model)

            w = soft_layer.model_intensity.layers[-1].get_weights()[0]
            b = soft_layer.model_intensity.layers[-1].get_weights()[1]
            print('bias from base_vae_model: ', b)
            print('-----------------------------------------------------------------')
            with open(FLAGS.base_vae_model + "_soft_layer.pkl", 'wb') as out:
                pickle.dump({'w': w, 'b': b}, out, protocol=2)
                out.close()

            with tf.variable_scope("model", reuse=True) as scope:
                scope.reuse_variables()
                b1 = tf.get_variable("b1", [FLAGS.num_au, 2]).assign(np.array(b))
                w1 = tf.get_variable("w1", [300, FLAGS.num_au, 2]).assign(np.array(w))
                sess.run(b1)
                sess.run(w1)
            print("uploaded bias from base_vae_model: ", sess.run('model/b1:0'))
        elif FLAGS.keep_train_dir:
            print('checkpoint dir: ', FLAGS.keep_train_dir)
            model_file = tf.train.latest_checkpoint(FLAGS.keep_train_dir)

            if FLAGS.test_iter > 0:
                files = os.listdir(model_file[:model_file.index('model')])
                if 'model' + str(FLAGS.test_iter) + '.index' in files:
                    model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
                    print(">>>> model_file2: ", model_file)

            print("--- Restoring model weights from " + model_file)
            saver.restore(sess, model_file)
            print("uploaded bias from MAML ckpt: ", sess.run('model/b1:0'))
    print("================================================================================")

    if FLAGS.adaptation:
        print("ADAPTATION")
        print("before adaptation w: ", sess.run('model/w1:0')[0])
        print("before adaptation bias: ", sess.run('model/b1:0'))
        print("================================================================================")
        test(model, sess, trained_model_dir, data_generator, all_used_frame_set)
    else:
        print("TRAIN")
        print("================================================================================")
        train(model, data_generator, saver, sess, trained_model_dir, resume_itr)


    end_time = datetime.now()
    elapse = end_time - start_time
    print("================================================================================")
    print(">>>>>> elapse time: " + str(elapse))
    print("================================================================================")


if __name__ == "__main__":
    main()
