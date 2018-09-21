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
flags.DEFINE_bool('train_test', False, 're-train model with the test set')
flags.DEFINE_bool('train_test_inc', False, 're-train model increasingly')
flags.DEFINE_bool('test_test', False, 'test the test set with test-model')
flags.DEFINE_bool('test_train', False, 'test the test set with train-model')
# for train, train_test
flags.DEFINE_integer('sbjt_start_idx', 0, 'starting subject index')
# for test_test, test_train
flags.DEFINE_integer('num_test_tasks', 1, 'num of task for test')
flags.DEFINE_string('testset_dir', './data/1/', 'directory for test set')
flags.DEFINE_string('test_result_dir', 'robert', 'directory for test result log')
# for train_test, test_test
flags.DEFINE_string('keep_train_dir', None,
                    'directory to read already trained model when training the model again with test set')
flags.DEFINE_integer('local_subj', 0, 'local weight subject')
flags.DEFINE_integer('kshot_seed', 0, 'seed for k shot sampling')
flags.DEFINE_integer('weight_seed', 0, 'seed for initial weight')
flags.DEFINE_integer('num_au', 12, 'number of AUs used to make AE')
flags.DEFINE_integer('au_idx', 12, 'au index to deal with in the given vae model')
flags.DEFINE_string('vae_model', './model_au_12.h5', 'vae model dir from robert code')
flags.DEFINE_string('gpu', "0,1,2,3", 'vae model dir from robert code')
flags.DEFINE_bool('global_test', False, 'get test evaluation throughout all test tasks')
flags.DEFINE_bool('global_model', True, 'model is trained with all train/test tasks')

def train(model, saver, sess, trained_model_dir, metatrain_input_tensors, metaval_input_tensors, resume_itr=0):
    SUMMARY_INTERVAL = 500
    SAVE_INTERVAL = 5000

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
        # if FLAGS.train_test and (itr == FLAGS.metatrain_iterations):
        if (itr % SUMMARY_INTERVAL == 0) or (itr == 1) or (itr == FLAGS.metatrain_iterations):
            input_tensors.extend([model.fast_weights])

        # SUMMARY_INTERVAL 마다 accuracy 계산해둠
        if (itr % SUMMARY_INTERVAL == 0) or (itr == 1) or (itr == FLAGS.metatrain_iterations):
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
                    FLAGS.init_weight) + '.sbjt_' + str(FLAGS.sbjt_start_idx) + ':' + str(
                    FLAGS.meta_batch_size) + '.updatelr' + str(FLAGS.train_update_lr) + '.metalr' + str(
                    FLAGS.meta_lr) + '.numstep' + str(FLAGS.num_updates) + "/train"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                save_path = "./logs/result/train/" + trained_model_dir + "/"
                if FLAGS.train_test:
                    retrained_model_dir = 'sbjt' + str(FLAGS.sbjt_start_idx) + ':' + str(
                        FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(
                        FLAGS.num_updates) + '.updatelr' + str(
                        FLAGS.train_update_lr) + '.metalr' + str(FLAGS.meta_lr)
                    save_path += retrained_model_dir
                elif FLAGS.train_test_inc:
                    retrained_model_dir = 'inc.sbjt' + str(FLAGS.sbjt_start_idx) + ':' + str(
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

            # save weight norm
            local_w = result[1][0]
            local_b = result[1][1]
            global_w = sess.run('model/w1:0')
            global_b = sess.run('model/b1:0')

            w_arr = [global_w]
            b_arr = [global_b]
            for i in range(FLAGS.meta_batch_size):
                w_arr.append(local_w[i])
                b_arr.append(local_b[i])
            save_path = FLAGS.logdir + '/' + trained_model_dir + '/weight'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if FLAGS.train_test:
                out = open(save_path + "/test_" + str(itr) + ".pkl", 'wb')
            else:
                out = open(save_path + "/train_" + str(itr) + ".pkl", 'wb')
            pickle.dump({'w': w_arr, 'b': b_arr}, out, protocol=2)
            out.close()


        # SAVE_INTERVAL 마다 weight값 파일로 떨굼
        if (itr % SAVE_INTERVAL == 0) or (itr == FLAGS.metatrain_iterations):
            if FLAGS.train_test:
                retrained_model_dir = '/' + 'sbjt' + str(FLAGS.sbjt_start_idx) + ':' + str(
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
                        print('>>>>>> Local weights for subject: ', i, sess.run(model.weights['w1']),
                              sess.run('model/b1:0'))
                        print("-----------------------------------------------------------------")
                        if not os.path.exists(local_model_dir):
                            os.makedirs(local_model_dir)
                        saver.save(sess, local_model_dir + '/subject' + str(i))

            else:
                # to train the model increasingly whenever new test is coming.
                saver.save(sess, FLAGS.logdir + '/' + trained_model_dir + '/model' + str(itr))


def test_each_subject(w, b, sbjt_start_idx):  # In case when test the model with the whole rest frames
    from vae_model import VAE
    import EmoData as ED
    import cv2
    import pickle
    batch_size = 10
    vae_model = VAE((160, 240, 1), batch_size, 12)
    vae_model.loadWeight(FLAGS.vae_model, w, b)

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

    def get_y_hat(file_names):
        nb_samples = len(file_names)
        t0, t1 = 0, batch_size
        yhat = []
        while True:
            t1 = min(nb_samples, t1)
            file_names_batch = file_names[t0:t1]
            imgs = [cv2.imread(filename) for filename in file_names_batch]
            img_arr, pts, pts_raw = pp.batch_transform(imgs, preprocessing=True, augmentation=False)
            pred = vae_model.testWithSavedModel(img_arr)
            yhat.extend(pred)
            if t1 == nb_samples: break
            t0 += batch_size  # 작업한 배치 사이즈만큼 t0와 t1늘림
            t1 += batch_size
        return np.array(yhat)

    test_subjects = os.listdir(FLAGS.testset_dir)
    test_subjects.sort()

    test_subjects = test_subjects[sbjt_start_idx:sbjt_start_idx + 1]

    print("test_subjects: ", test_subjects)

    y_hat_all = []
    y_lab_all = []
    for test_subject in test_subjects:
        print("============> subject: ", test_subject)
        data = pickle.load(open(FLAGS.testset_dir + test_subject, "rb"), encoding='latin1')
        test_file_names = data['test_file_names']
        y_hat = get_y_hat(test_file_names)
    return y_hat, data['lab']


def test_all(w, b, trained_model_dir):  # In case when test the model with the whole rest frames
    from vae_model import VAE
    import EmoData as ED
    import cv2
    import pickle
    batch_size = 10
    vae_model = VAE((160, 240, 1), batch_size, 12)
    vae_model.loadWeight(FLAGS.vae_model, w, b)

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

    def get_y_hat(file_names):
        nb_samples = len(file_names)
        t0, t1 = 0, batch_size
        yhat = []
        while True:
            t1 = min(nb_samples, t1)
            file_names_batch = file_names[t0:t1]
            imgs = [cv2.imread(filename) for filename in file_names_batch]
            img_arr, pts, pts_raw = pp.batch_transform(imgs, preprocessing=True, augmentation=False)
            pred = vae_model.testWithSavedModel(img_arr)
            yhat.extend(pred)
            if t1 == nb_samples: break
            t0 += batch_size  # 작업한 배치 사이즈만큼 t0와 t1늘림
            t1 += batch_size
        return np.array(yhat)

    test_subjects = os.listdir(FLAGS.testset_dir)
    test_subjects.sort()
    test_subjects = test_subjects[FLAGS.sbjt_start_idx:FLAGS.sbjt_start_idx + FLAGS.num_test_tasks]
    print("test_subjects: ", test_subjects)

    for test_subject in test_subjects:
        print("============> subject: ", test_subject)
        data = pickle.load(open(FLAGS.testset_dir + test_subject, "rb"), encoding='latin1')
        test_file_names = data['test_file_names']
        y_hat = get_y_hat(test_file_names)
        save_path = "./logs/result/test_test/" + trained_model_dir
        if FLAGS.test_train:
            save_path = "./logs/result/test_train/" + trained_model_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print_summary(y_hat, data['lab'],
                      log_dir=save_path + "/" + test_subject.split(".")[0] + ".txt")




def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

    if FLAGS.test_train or FLAGS.test_test:
        temp_kshot = FLAGS.update_batch_size
        FLAGS.update_batch_size = 1
    data_generator = DataGenerator()

    dim_output = data_generator.num_classes
    dim_input = data_generator.dim_input

    if FLAGS.train:  # only construct training model if needed
        print("===================================1")

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
        print("===================================3")
        model.construct_model(input_tensors=metatrain_input_tensors, prefix='metatrain_')
    else:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=20)

    sess = tf.InteractiveSession()

    if FLAGS.test_train or FLAGS.test_test:
        FLAGS.update_batch_size = temp_kshot

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
    elif FLAGS.local_subj > 0:
        trained_model_dir = FLAGS.keep_train_dir

    print(">>>>> trained_model_dir: ", FLAGS.logdir + '/' + trained_model_dir)

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
    if FLAGS.init_weight and FLAGS.train:
        model.weights['w1'].load(pred_weights[0], sess)
        model.weights['b1'].load(pred_weights[1], sess)
    print('updated weights from vae?: ', FLAGS.init_weight, sess.run(model.weights['w1']), sess.run('model/b1:0'))
    print("========================================================================================")

    ################## Test ##################
    if FLAGS.test_train or FLAGS.test_test:
        def process(sbjt_start_idx):
            if FLAGS.test_test:
                if FLAGS.global_model:  # 모델이 모든 train or test tasks로 학습된거기때문에 항상 0부터 meta_batch_size까지 이용해서 구해진거가됨
                    trained_model_dir = FLAGS.keep_train_dir + '/' + 'sbjt' + str(0) + ':' + str(
                        FLAGS.meta_batch_size) + '.ubs_' + str(
                        FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(
                        FLAGS.train_update_lr) + '.metalr' + str(FLAGS.meta_lr)
                else:
                    trained_model_dir = FLAGS.keep_train_dir + '/' + 'sbjt' + str(sbjt_start_idx) + ':' + str(
                        FLAGS.meta_batch_size) + '.ubs_' + str(
                        FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(
                        FLAGS.train_update_lr) + '.metalr' + str(FLAGS.meta_lr)
            else:
                trained_model_dir = 'cls_' + str(FLAGS.num_classes) + '.mbs_' + str(
                    FLAGS.meta_batch_size) + '.ubs_' + str(
                    FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(
                    FLAGS.train_update_lr) + '.metalr' + str(FLAGS.meta_lr) + '.initweight' + str(FLAGS.init_weight)
            all_au = ['au1', 'au2', 'au4', 'au5', 'au6', 'au9', 'au12', 'au15', 'au17', 'au20', 'au25', 'au26']
            # all_au = ['au12'] * 12
            w_arr = None
            b_arr = None
            for au in all_au:
                model_file = None
                print('--------- model file dir: ', FLAGS.logdir + '/' + au + '/' + trained_model_dir)
                model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + au + '/' + trained_model_dir)
                print(">>>> model_file from ", au, ": ", model_file)
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
                            print(">>>> model_file2: ", model_file)
                        else:
                            print(" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", files)
                    print("Restoring model weights from " + model_file)
                    saver.restore(sess, model_file)
                    w = sess.run('model/w1:0')
                    b = sess.run('model/b1:0')
                    if w_arr is None:
                        w_arr = w
                        b_arr = b
                    else:
                        w_arr = np.hstack((w_arr, w))
                        b_arr = np.vstack((b_arr, b))
                    print("updated weights from ckpt: ", w, b)
                    print('----------------------------------------------------------')
            return test_each_subject(w_arr, b_arr, sbjt_start_idx)

        if FLAGS.global_test:
            print(
                "<<<<<<<<<<<< want to see the evaluation by concatenating all subject's predictions >>>>>>>>>>>>>>>>>")
            save_path = "./logs/result/test_test/" + trained_model_dir
            if FLAGS.test_train:
                save_path = "./logs/result/test_train/" + trained_model_dir
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            y_hat = []
            y_lab = []
            for i in range(FLAGS.sbjt_start_idx, FLAGS.num_test_tasks):
                result = process(i)
                y_hat.append(result[0])
                y_lab.append(result[1])
                print("y_hat shape:", result[0].shape)
                print("y_lab shape:", result[1].shape)
                print(">> y_hat_all shape:", np.vstack(y_hat).shape)
                print(">> y_lab_all shape:", np.vstack(y_lab).shape)
            print_summary(np.vstack(y_hat), np.vstack(y_lab), log_dir=save_path + "/" + "test.txt")

        else:
            print("<<<<<<<<<<<< model was trained using all test/train tasks >>>>>>>>>>>>>>>>>")
            all_au = ['au1', 'au2', 'au4', 'au5', 'au6', 'au9', 'au12', 'au15', 'au17', 'au20', 'au25', 'au26']
            if FLAGS.test_test:
                trained_model_dir = FLAGS.keep_train_dir + '/' + 'sbjt' + str(FLAGS.sbjt_start_idx) + ':' + str(
                    FLAGS.meta_batch_size) + '.ubs_' + str(
                    FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(
                    FLAGS.train_update_lr) + '.metalr' + str(FLAGS.meta_lr)
            w_arr = None
            b_arr = None
            for au in all_au:
                model_file = None
                model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + au + '/' + trained_model_dir)
                print(">>>> model_file from ", au, ": ", model_file)
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
                            print(">>>> model_file2: ", model_file)
                        else:
                            print(" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", files)
                    print("Restoring model weights from " + model_file)
                    saver.restore(sess, model_file)
                    w = sess.run('model/w1:0')
                    b = sess.run('model/b1:0')
                    if w_arr is None:
                        w_arr = w
                        b_arr = b
                    else:
                        w_arr = np.hstack((w_arr, w))
                        b_arr = np.vstack((b_arr, b))
                    print("updated weights from ckpt: ", w, b)
                    print('----------------------------------------------------------')

            if FLAGS.init_weight and (model_file == None):
                w_arr = None
                b_arr = None
                print(">>>>>>>>>> test robert's model ")
            test_all(w_arr, b_arr, trained_model_dir)




    ################## Train ##################

    # train_train or train_test
    elif FLAGS.resume:  # 디폴트로 resume은 항상 true. 따라서 train중간부터 항상 시작 가능.
        model_file = None

        if FLAGS.train_test:
            tmp_trained_model_dir = trained_model_dir + '/' + 'sbjt' + str(FLAGS.sbjt_start_idx) + ':' + str(
                FLAGS.meta_batch_size) + '.ubs_' + str(
                FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(
                FLAGS.train_update_lr) + '.metalr' + str(FLAGS.meta_lr)
            model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + tmp_trained_model_dir)
        else:
            model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + trained_model_dir)

        print(">>>>> trained_model_dir: ", FLAGS.logdir + '/' + trained_model_dir)

        w = None
        b = None
        print(">>>> model_file1: ", model_file)

        #TODO delete this if
        if FLAGS.local_subj > 0:
            model_file = model_file[:model_file.index('subject')] + 'subject' + str(FLAGS.local_subj - 14)
            print(">>>> model_file2: ", model_file)
        if model_file:
            if FLAGS.train_test:
                if FLAGS.test_iter > 0:
                    files = os.listdir(model_file[:model_file.index('model')])
                    if 'model' + str(FLAGS.test_iter) + '.index' in files:
                        model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
                        print(">>>> model_file2: ", model_file)
            print("1. Restoring model weights from " + model_file)
            saver.restore(sess, model_file)
            w = sess.run('model/w1:0').tolist()
            b = sess.run('model/b1:0').tolist()
            print("updated weights from ckpt: ", np.array(w), np.array(b))
            if not FLAGS.test_test:
                ind1 = model_file.index('model')
                resume_itr = int(model_file[ind1 + 5:])
                print('resume_itr: ', resume_itr)
                # else:
                #     model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + trained_model_dir)
                #     print("2. Restoring model weights from " + model_file)
                #     saver.restore(sess, model_file)
                #     w = sess.run('model/w1:0').tolist()
                #     b = sess.run('model/b1:0').tolist()
                #     print("updated weights from ckpt: ", np.array(w), np.array(b))
                #     if not FLAGS.test_test:
                #         ind1 = model_file.index('model')
                #         if FLAGS.train_test: resume_itr = 0
                #         else: resume_itr = int(model_file[ind1 + 5:])
                #         print('resume_itr: ', resume_itr)
    else:
        model_file = None
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + trained_model_dir)
        if model_file:
            print("2. Restoring model weights from " + model_file)
            saver.restore(sess, model_file)
            w = sess.run('model/w1:0').tolist()
            b = sess.run('model/b1:0').tolist()
            print("updated weights from ckpt: ", np.array(w), np.array(b))

    print("=====================================================================================")

    if FLAGS.train:
            train(model, saver, sess, trained_model_dir, metatrain_input_tensors, metaval_input_tensors, resume_itr)
    end_time = datetime.now()
    elapse = end_time - start_time
    print("=======================================================")
    print(">>>>>> elapse time: " + str(elapse))
    print("=======================================================")


if __name__ == "__main__":
    main()
