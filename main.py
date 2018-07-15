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

flags.DEFINE_bool('init_weight', True, 'Initialize weights from the base model')
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


def train(model, saver, sess, trained_model_dir, metatrain_input_tensors, metaval_input_tensors, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 500

    if FLAGS.train_test:
        resume_itr = 0

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + trained_model_dir, sess.graph)

    feed_dict = {model.inputa: metatrain_input_tensors['inputa'].eval(),
                 model.inputb: metatrain_input_tensors['inputb'].eval(),
                 model.labela: metatrain_input_tensors['labela'].eval(),
                 model.labelb: metatrain_input_tensors['labelb'].eval(), model.meta_lr: FLAGS.meta_lr}

    print('Done initializing, starting training.')

    for itr in range(resume_itr + 1, FLAGS.metatrain_iterations + 1):

        input_tensors = [model.metatrain_op]

        # when train the model again with the test set, local weight needs to be saved at the last iteration.
        if FLAGS.train_test and (itr == FLAGS.metatrain_iterations):
            input_tensors.extend([model.fast_weights])

        # SUMMARY_INTERVAL 마다 accuracy 계산해둠
        if (itr % SUMMARY_INTERVAL) == 0:
            input_tensors.extend([model.result1, model.result2])

        result = sess.run(input_tensors, feed_dict)


        # SUMMARY_INTERVAL 마다 accuracy 쌓아둠
        if itr % SUMMARY_INTERVAL == 0:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>summary")

            # run for the validation
            feed_dict_val = {model.inputa: metaval_input_tensors['inputa'].eval(),
                         model.inputb: metaval_input_tensors['inputb'].eval(),
                         model.labela: metaval_input_tensors['labela'].eval(),
                         model.labelb: metaval_input_tensors['labelb'].eval(), model.meta_lr: 0}
            result_val = sess.run(input_tensors, feed_dict_val)

            def summary(maml_result, set):
                print(set)
                print_str = 'Iteration ' + str(itr)
                print(print_str)
                y_hata = np.vstack(np.array(maml_result[-2][0]))  # length = num_of_task * N * K
                y_laba = np.vstack(np.array(maml_result[-2][1]))
                save_path = "./logs/result/" + str(FLAGS.update_batch_size) + "shot/" + 'weight' + str(
                    FLAGS.init_weight) + '.sbjt_' + str(FLAGS.train_start_idx) + ':' + str(
                    FLAGS.meta_batch_size) + '.updatelr' + str(FLAGS.train_update_lr) + '.metalr' + str(
                    FLAGS.meta_lr) + '.numstep' + str(FLAGS.num_updates) + "/train"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                save_path = "./logs/result/train/" + trained_model_dir + "/"
                if FLAGS.train_test:
                    retrained_model_dir = 'sbjt' + str(FLAGS.train_start_idx) + ':' + str(
                        FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(
                        FLAGS.num_updates) + '.updatelr' + str(
                        FLAGS.train_update_lr) + '.metalr' + str(FLAGS.meta_lr)
                    save_path += retrained_model_dir
                elif FLAGS.train_test_inc:
                    retrained_model_dir = 'inc.sbjt' + str(FLAGS.train_start_idx) + ':' + str(
                        FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(
                        FLAGS.num_updates) + '.updatelr' + str(
                        FLAGS.train_update_lr) + '.metalr' + str(FLAGS.meta_lr)
                    save_path += retrained_model_dir
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                print_summary(y_hata, y_laba, log_dir=save_path + "/outa_" + set + "_" + str(itr) + ".txt")
                print("------------------------------------------------------------------------------------")
                recent_y_hatb = np.array(maml_result[-1][0][
                                             FLAGS.num_updates - 1])  # 모든 num_updates별 outb, labelb말고 가장 마지막 update된 outb, labelb만 가져오면됨. 14 tasks가 병렬계산된 값이므로  length = num_of_task * N * K
                y_hatb = np.vstack(recent_y_hatb)
                recent_y_labb = np.array(maml_result[-1][1][FLAGS.num_updates - 1])
                y_labb = np.vstack(recent_y_labb)
                print_summary(y_hatb, y_labb, log_dir=save_path + "/outb_" + set + "_" + str(itr) + ".txt")
                print("====================================================================================")


            summary(result, "TR")
            summary(result_val, "TE")

        # SAVE_INTERVAL 마다 weight값 파일로 떨굼
        if (itr % SAVE_INTERVAL == 0) or (itr == FLAGS.metatrain_iterations):
            if FLAGS.train_test:
                retrained_model_dir = '/' + 'sbjt' + str(FLAGS.train_start_idx) + ':' + str(
                    FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(
                    FLAGS.num_updates) + '.updatelr' + str(
                    FLAGS.train_update_lr) + '.metalr' + str(FLAGS.meta_lr)

                save_path = FLAGS.logdir + '/' + trained_model_dir + retrained_model_dir
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                saver.save(sess, save_path + '/model' + str(itr))

                # save local weight at the last iteration
                if itr == FLAGS.metatrain_iterations:
                    print(">>>>>>>>>>>>>> local save !! : ", itr)
                    local_w = result[1][0]
                    local_b = result[1][1]
                    print("========================================================================================")
                    print('>>>>>> Global weights: ', sess.run(model.weights['w1']), sess.run('model/b1:0'))
                    local_model_dir = save_path + '/local'
                    for i in range(FLAGS.meta_batch_size):
                        model.weights['w1'].load(local_w[i], sess)
                        model.weights['b1'].load(local_b[i], sess)
                        print('>>>>>> Local weights for subject: ', i, sess.run(model.weights['w1']), sess.run('model/b1:0'))
                        print("-----------------------------------------------------------------")
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        saver.save(sess, local_model_dir + '/subject' + str(i))

            else:
                # to train the model increasingly whenever new test is coming.
                saver.save(sess, FLAGS.logdir + '/' + trained_model_dir + '/model' + str(itr))







def test(model, saver, sess, trained_model_dir, data_generator):
    result_arr = []
    NUM_TEST_POINTS = FLAGS.num_test_pts
    print("===========================================================================")

    for _ in range(NUM_TEST_POINTS):
        feed_dict = {model.meta_lr: 0.0}  # do not optimize in test because it needs to be iterated.
        input_tensor = [model.metaval_result1, model.metaval_result2]
        result = sess.run(input_tensor, feed_dict)
        result_arr.append(result)

    save_path = "./logs/result/" + str(FLAGS.train_update_batch_size) + "shot/" + 'weight' + str(
        FLAGS.init_weight) + '.updatelr' + str(FLAGS.train_update_lr) + '.metalr' + str(
        FLAGS.meta_lr) + '.numstep' + str(FLAGS.num_updates) + "/test/" + str(FLAGS.test_iter)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    i = 0
    for result in result_arr:
        y_hata = np.array(result[0][0])[
            0]  # result[0][0]=y_hata: has shape (1,2,1,2)=(num.of.task, 2*k, num.of.au, one-hot label); test task는 항상 1개니까 0인덱스만 불러와도 상관없음
        y_laba = np.array(result[0][1])[0]

        y_hatb = result[1][0][FLAGS.num_updates - 1][
            0]  # result[1][0]=y_hat: has (num_updates) elts. We see only the recent elt.==>result[1][0][FLAGS.num_updates-1]: has shape (1,2,1,2)=(num.of.task, 2*k, num.of.au, one-hot label)
        y_labb = result[1][1][FLAGS.num_updates - 1][0]

        print_summary(y_hata, y_laba, log_dir=save_path + "/outa_" + str(FLAGS.subject_idx) + ".iter" + str(i) + ".txt")
        print("------------------------------------------------------------------------------------")

        print_summary(y_hatb, y_labb, log_dir=save_path + "/outb_" + str(FLAGS.subject_idx) + ".iter" + str(i) + ".txt")
        print("====================================================================================")
        i += 1


def test_test(w, b, trained_model_dir):  # In case when test the model with the whole rest frames
    from vae_model import VAE
    import EmoData as ED
    import cv2
    import pickle
    batch_size = 10
    vae_model = VAE((160, 240, 1), batch_size)
    vae_model.loadWeight("./model_log300.h5", w, b)

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

    def predict_label(file_path):
        imgs = []
        for filename in file_path:
            img = cv2.imread(filename)
            imgs.append(img)
        img_arr, pts, pts_raw = pp.batch_transform(imgs, preprocessing=True, augmentation=False)
        return vae_model.testWithSavedModel(img_arr)

    def get_y_hat(test_file_names, N_batch):
        file_names_batch = np.reshape(test_file_names[:N_batch * batch_size], [N_batch, batch_size])

        yhat_arr = []
        for file_path in file_names_batch:
            pred = predict_label(file_path)
            yhat_arr.extend(pred)
        if N_batch * batch_size != len(data['test_file_names']):
            rest_pred = predict_label(test_file_names[N_batch * batch_size:])
            yhat_arr.extend(rest_pred)
        return np.array(yhat_arr)

    test_subjects = os.listdir(FLAGS.testset_dir)
    test_subjects.sort()
    test_subjects = test_subjects[FLAGS.test_start_idx - 14:FLAGS.test_start_idx - 14 + FLAGS.test_num]
    print("test_subjects: ", test_subjects)

    for test_subject in test_subjects:
        data = pickle.load(open(FLAGS.testset_dir + test_subject, "rb"), encoding='latin1')

        N_batch = int(len(data['test_file_names']) / batch_size)

        print(test_subject.split(".")[0], " total len:", len(data['test_file_names']))
        y_hat = get_y_hat(data['test_file_names'], N_batch)
        print(y_hat.shape)
        save_path = "./logs/result/test_test/" + trained_model_dir
        if FLAGS.test_train:
            save_path = "./logs/result/test_train/" + trained_model_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print_summary(y_hat, data['y_lab'], log_dir=save_path + "/" + test_subject.split(".")[0] + ".txt")


def main():
    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

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

    if not FLAGS.train:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    trained_model_dir = 'cls_' + str(FLAGS.num_classes) + '.mbs_' + str(FLAGS.meta_batch_size) + '.ubs_' + str(
        FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(
        FLAGS.train_update_lr) + '.metalr' + str(FLAGS.meta_lr) + '.initweight' + str(FLAGS.init_weight)
    if FLAGS.train_test or FLAGS.train_test_inc:
        trained_model_dir = FLAGS.keep_train_dir  # TODO: model0이 없는 경우 keep_train_dir에서 model을 subject경로로 옮기고 그 모델의 인덱스를 0으로 만드는 작업해주기.
    elif FLAGS.test_test:
        trained_model_dir = FLAGS.keep_train_dir
        trained_model_dir += '/' + 'sbjt' + str(FLAGS.test_start_idx) + ':' + str(FLAGS.test_num) + '.ubs_' + str(
            FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(
            FLAGS.train_update_lr) + '.metalr' + str(FLAGS.meta_lr)

    print(">>>>> trained_model_dir: ", trained_model_dir)

    # if FLAGS.stop_grad:
    #     trained_model_dir += 'stopgrad'
    # if FLAGS.baseline:
    #     trained_model_dir += FLAGS.baseline
    # else:
    #     print('Norm setting not recognized.')


    resume_itr = 0

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    print("========================================================================================")
    print('initial weights: ', sess.run(model.weights['w1']), sess.run('model/b1:0'))
    print('weights from vae : ', pred_weights)
    if FLAGS.init_weight:
        model.weights['w1'].load(pred_weights[0], sess)
        model.weights['b1'].load(pred_weights[1], sess)
    print('updated weights from vae?: ', FLAGS.init_weight, sess.run(model.weights['w1']), sess.run('model/b1:0'))
    print("========================================================================================")

    if FLAGS.resume or not FLAGS.train:
        model_file = None
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + trained_model_dir)
        w = None
        b = None
        print(">>>> model_file1: ", model_file)
        # model_file = tf.train.latest_checkpoint('../model.h5')
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
            print(">>>> model_file2: ", model_file)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1 + 5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)
            w = sess.run('model/w1:0').tolist()
            b = sess.run('model/b1:0').tolist()
            print("updated weights from ckpt: ", np.array(w), np.array(b))
            print('resume_itr: ', resume_itr)
            print("=====================================================================================")

    if FLAGS.test_test or FLAGS.test_train:
        test_test(w, b, trained_model_dir)
    else:
        if FLAGS.train:
            train(model, saver, sess, trained_model_dir, metatrain_input_tensors, metaval_input_tensors, resume_itr)
        else:
            test(model, saver, sess, trained_model_dir, data_generator)
    end_time = datetime.now()
    elapse = end_time - start_time
    print("=======================================================")
    print(">>>>>> elapse time: " + str(elapse))
    print("=======================================================")


if __name__ == "__main__":
    main()
