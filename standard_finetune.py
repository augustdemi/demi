import EmoEstimator as EE
import EmoData as ED
import keras.backend as KB
from keras.layers import Dense, Lambda, Input, Reshape
import keras as K
import numpy as np
import argparse
from datetime import datetime
import os
from feature_layers import feature_layer

start_time = datetime.now()

parser = argparse.ArgumentParser(description='extract feace images from raw datasets')
parser.add_argument("-i", "--input", type=str, default='init', help="files created from GP")
parser.add_argument("-o", "--output", type=str, default='./model_output/disfa_all', help="files creaded from VAE")
parser.add_argument("-n", "--nb_iter", type=int, default=1, help="number of VAE iterations")
parser.add_argument("-w", "--warming", type=int, default=1, help="factor on kl loss")

parser.add_argument("-log", "--log_dir", type=str, default='default', help="log dir")
parser.add_argument("-au", "--au_index", type=int, default=8, help="au index")
parser.add_argument("-num_au", "--num_au", type=int, default=8, help="number of au to make the model previously.")
parser.add_argument("-e", "--init_epoch", type=int, default=0, help="Epoch at which to start training")
parser.add_argument("-g", "--gpu", type=str, default='0,1,2,3', help="files created from GP")
parser.add_argument("-rm", "--restored_model", type=str, default='', help="already trianed model to restore")
parser.add_argument("-sm", "--saving_model", type=str, default='', help="model name to save")
parser.add_argument("-lr", "--lr", type=float, default=1.0, help="learning rate")
parser.add_argument("-bal", "--balance", type=bool, default=False, help="Make the dataset balanced or not")
parser.add_argument("-kshot", "--kshot", type=int, default=0, help="test kshot learning")
parser.add_argument("-mbs", "--meta_batch_size", type=int, default=13, help="num of task to use for kshot learning")
parser.add_argument("-sidx", "--subject_index", type=int, default=0, help="test subject index")
parser.add_argument("-kshot_seed", "--kshot_seed", type=int, default=0, help="kshot seed")
parser.add_argument("-feat_path", "--feat_path", type=str, default='', help="extracted feature csv path")
parser.add_argument("-logdir", "--logdir", type=str, default='', help="extracted feature csv path")
parser.add_argument("-tmd", "--trained_model_dir", type=str, default='', help="extracted feature csv path")
parser.add_argument("-test", "--test", type=bool, default=True, help="adapt to test data")
parser.add_argument("-op", "--opti", type=str, default="adam", help="name of optimizer")
parser.add_argument("-maml", "--maml_model", type=str, default="", help="name of maml_model")
parser.add_argument("-m_iter", "--m_iter", type=str, default="", help="num of iter used when training maml model")
parser.add_argument("-used_frame", "--used_frame_info_path", type=str, default="",
                    help="path for the csv file of the used frames in MAML adaptation ")
parser.add_argument("-label_path", "--label_path", type=str, default="", help="path for label")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# lins input
TOTAL_AU = 8

source_data = args.input
nb_iter = args.nb_iter
au_index = args.au_index
print("======================================= au_index: ", au_index)
if args.saving_model == '':
    model_name = './' + args.restored_model + '.h5'
else:
    model_name = './' + args.saving_model + '.h5'

batch_size = 20  # dont change it!
log_dir_model = './model'

TR, batch_size = ED.provider_back.flow_from_kshot_csv(args.used_frame_info_path, args.feat_path, args.label_path,
                                          args.subject_index)

TE, batch_size = ED.provider_back.flow_from_kshot_csv(args.used_frame_info_path, args.feat_path, args.label_path,
                                          args.subject_index, eval=True, padding='same', batch_size=batch_size)


def generator(dat_dict):
    while True:
        feature = next(dat_dict['feat'])
        lab = next(dat_dict['lab'])
        if lab.shape[1] == TOTAL_AU and au_index < TOTAL_AU:
            lab = lab[:, au_index]
            lab = np.reshape(lab, (lab.shape[0], 1, lab.shape[1]))
        yield [feature], [lab]



GEN_TR = generator(TR)  # train data안의 그룹 별로 (img/label이 그룹인듯) 정해진 배치사이즈만큼의 배치 이미지 혹은 배치 라벨을 생성
GEN_TE = generator(TE)


def pred_loss(y_true, y_pred):
    # ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    ce = EE.losses.categorical_crossentropy(y_true, y_pred)
    return ce


sum_vac_disfa_dir = log_dir_model + '/z_val/disfa/' + args.log_dir
sum_mult_out_dir = 'res_disfa_' + str(args.warming).zfill(4) + '.csv/' + args.log_dir

three_layers = feature_layer(batch_size, args.num_au)

import pickle


aus = ['au1', 'au2', 'au4', 'au6', 'au9', 'au12', 'au25', 'au26']

if args.logdir != '':
    print('---------------- load from MAML')
    if args.trained_model_dir != '':
        data = pickle.load(
            open(args.logdir + '/' + args.trained_model_dir + "/soft_weights" + args.m_iter + ".pkl", "rb"),
            encoding='latin1')
    else:
        data = pickle.load(open(args.logdir + 'm1.' + aus[args.au_index] + '.alpha0.05_beta0.05.pkl', "rb"), encoding='latin1')

    w = np.array(data['w1'])
    b = np.array(data['b1'])
    print(w.shape)
    print(w)
    print('--------------------------------')
    three_layers.loadWeight_pkl(args.restored_model, w, b)
else:
    three_layers.loadWeight(args.restored_model)

model_intensity = three_layers.model_intensity

for i in range(len(model_intensity.layers) - 1):
    model_intensity.layers[i].trainable = False
for i in range(len(model_intensity.layers)):
    print(model_intensity.layers[i], model_intensity.layers[i].trainable)

if not os.path.exists(sum_vac_disfa_dir):
    os.makedirs(sum_vac_disfa_dir)


if args.opti.startswith('adam'):
    model_intensity.compile(optimizer=K.optimizers.Adam(lr=args.lr), loss=pred_loss)
else:
    model_intensity.compile(optimizer=K.optimizers.Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.0), loss=pred_loss)


model_intensity.summary()
print('loaded softmax weight of model_intensity: ', model_intensity.layers[-1].get_weights()[0])
print('loaded softmax weight of model_intensity: ', model_intensity.layers[-1].get_weights()[1])

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

model_intensity.fit_generator(
    generator=GEN_TR,
    # samples_per_epoch=960,  # number of samples to process before going to the next epoch.
    samples_per_epoch=batch_size,  # number of samples to process before going to the next epoch.
    # validation_data=GEN_TE,  # integer, total number of iterations on the data.
    nb_val_samples=5000,  # number of samples to use from validation generator at the end of every epoch.
    initial_epoch=args.init_epoch,
    nb_epoch=nb_iter,
    max_q_size=4,
    callbacks=[
        # early_stopping,
        EE.callbacks.summary_multi_output(
            gen_list=(generator(TR), generator(TE)),
            predictor=model_intensity.predict,  # predicted lable만을 예측, 이때는 augmented 되지 않은 train data를 이용하기 위해 분리?
            batch_size=batch_size,
            title=['TR', 'TE'],
            one_hot=True,
            log_dir=sum_mult_out_dir,
        )
    ]
)

if nb_iter > 0: model_intensity.save_weights(model_name)

end_time = datetime.now()
elapse = end_time - start_time
print("=======================================================")
print(">>>>>> elapse time: " + str(elapse))
print("=======================================================")
