import EmoEstimator as EE
import EmoData as ED
import keras.backend as KB
from keras.layers import Dense, Lambda, Input, Reshape
import keras as K
import numpy as np
import argparse
from datetime import datetime
import os
from vae_model import VAE

start_time = datetime.now()

parser = argparse.ArgumentParser(description='extract feace images from raw datasets')
parser.add_argument("-i", "--input", type=str, default='init', help="files created from GP")
parser.add_argument("-o","--output", type=str, default='./model_output/disfa_all', help="files creaded from VAE")
parser.add_argument("-n","--nb_iter",type=int, default=1, help="number of VAE iterations")
parser.add_argument("-w","--warming",type=int, default=1, help="factor on kl loss")
parser.add_argument("-tr", "--training_data", type=str, default='/home/mihee/dev/project/robert_data/test.h5',
                    help="path to training data set")
parser.add_argument("-te", "--test_data", type=str, default='/home/mihee/dev/project/robert_data/test.h5',
                    help="path to test data set")
parser.add_argument("-log", "--log_dir", type=str, default='default', help="log dir")
parser.add_argument("-dec", "--decoder", type=bool, default=True, help="train decoder layer or not")
parser.add_argument("-au", "--au_index", type=int, default=12, help="au index")
parser.add_argument("-num_au", "--num_au", type=int, default=12, help="number of au to make the model previously.")
parser.add_argument("-e", "--init_epoch", type=int, default=0, help="Epoch at which to start training")
parser.add_argument("-g", "--gpu", type=str, default='0,1,2,3', help="files created from GP")
parser.add_argument("-rm", "--restored_model", type=str, default='', help="already trianed model to restore")
parser.add_argument("-sm", "--saving_model", type=str, default='', help="model name to save")
parser.add_argument("-f", "--fine_tune", type=bool, default=False, help="if want to fine tune, gives True")
parser.add_argument("-lr", "--lr", type=float, default=1.0, help="learning rate")
parser.add_argument("-bal", "--balance", type=bool, default=False, help="Make the dataset balanced or not")
parser.add_argument("-kshot", "--kshot", type=int, default=0, help="test kshot learning")
parser.add_argument("-mbs", "--meta_batch_size", type=int, default=13, help="num of task to use for kshot learning")
parser.add_argument("-sidx", "--start_idx", type=int, default=0, help="start idx of task to use for kshot learning")
parser.add_argument("-deep", "--deep_feature", type=str, default='', help="dir to save the extracted deep feature")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# lins input
source_data = args.input
nb_iter = args.nb_iter
au_index = args.au_index
print("======================================= au_index: ", au_index)
if args.saving_model == '':
    model_name = './' + args.restored_model + '.h5'
else:
    model_name = './' + args.saving_model + '.h5'

batch_size = 32  # dont change it!
log_dir_model = './model'
latent_dim1 = 2048
latent_dim2 = 500
latent_dim3 = 300
w_1 = args.warming / 50

if args.kshot > 0:
    TR = ED.provider_back.flow_from_folder_kshot(args.training_data, batch_size, padding='same',
                                                 sbjt_start_idx=args.start_idx,
                                                 meta_batch_size=args.meta_batch_size, update_batch_size=args.kshot)
elif args.balance and au_index < 12:
    TR = ED.provider_back.flow_from_hdf5(args.training_data, batch_size, padding='same', au_idx=au_index)
else:
    TR = ED.provider_back.flow_from_hdf5(args.training_data, batch_size, padding='same')

TE = ED.provider_back.flow_from_hdf5(args.test_data, batch_size, padding='same')


pp = ED.image_pipeline.FACE_pipeline(
        histogram_normalization=True,
        grayscale=True,
        resize=True,
        rotation_range = 3,
        width_shift_range = 0.03,
        height_shift_range = 0.03,
        zoom_range = 0.03,
        random_flip = True,
        )


def generator(dat_dict, aug, mod=0, s=False):
    while True:

        img = next(dat_dict['img'])
        lab = next(dat_dict['lab'])
        sub = next(dat_dict['sub'])
        img, pts, pts_raw = pp.batch_transform(
                img,
                preprocessing=True,
                augmentation=aug)
        # lab = lab.argmax(2)
        if lab.shape[1] == 12 and au_index < 12:
            lab = lab[:, au_index]
            lab = np.reshape(lab, (lab.shape[0], 1, lab.shape[1]))
        else:
            lab = lab
        if mod==1:
            if(s): yield [img], [lab], [sub]
            else: yield [img], [lab]
        else:
            if(s): yield [img], [img, lab, img], [sub]
            else: yield [img], [img, lab, img]


GEN_TR = generator(TR, True) # train data안의 그룹 별로 (img/label이 그룹인듯) 정해진 배치사이즈만큼의 배치 이미지 혹은 배치 라벨을 생성
GEN_TE = generator(TE, False)

# X,Y,C dimenstion # ========================== start ==========================
X, Y = next(GEN_TR) # train data의 X = img batches , y = [img, lab, img]
inp_0_shape = X[0].shape[1:]
out_0_shape = Y[1].shape[1:]

print("inp_0_shape", inp_0_shape)
print("out_0_shape", out_0_shape)
inp_0       = Input(shape=inp_0_shape) # 케라스의 텐서 선언 ???????? input shape만을 위해 X를 사용하고 데이터 사용 더이상 안함? 텐서는 사이즈만으로 뭘함?
emb, shape = EE.networks.encoder(inp_0, norm=1)

from numpy import prod
from keras.layers import Dropout
print("shape before flatten: ", shape)
print("shape after flatten: ", emb.get_shape())
n_feat = prod(shape)

emb = Dropout(0.5)(emb)

latent_feat = Dense(latent_dim1, activation='relu', name='latent_feat')(emb)  # into 2048
intermediate = Dense(latent_dim2, activation='relu', name='intermediate')(latent_feat)  # into 500
z_mean = Dense(latent_dim3, name='z_mean')(intermediate)  # into latent_dim = 300은. output space의 dim이 될것.
z_log_sigma = Dense(latent_dim3)(intermediate)
print('==============================')
print('emb', emb.shape)
print('latent_feat', latent_feat.shape)
print('intermediate', intermediate.shape)
print('z_mean', z_mean.shape)
print('z_log_sigma', z_log_sigma.shape)

def sampling(args): ########### input param의 평균과 분산에 noise(target_mean, sd 기준)가 섞인 샘플링 값을줌
    z_mean, z_log_sigma = args
    # batch_size = 32
    epsilon = []
    for m, s in zip(np.zeros(latent_dim3), np.ones(latent_dim3)):
        epsilon.append(KB.random_normal(shape=[batch_size, 1], mean=m, std=s))
    epsilon = KB.concatenate(epsilon, 1)
    return z_mean + KB.exp(z_log_sigma) * epsilon


z = Lambda(sampling, output_shape=(latent_dim3,))([z_mean, z_log_sigma])  # 발굴한 feature space에다 노이즈까지 섞어서 샘플링한 z

out_1 = EE.layers.softmaxPDF(out_0_shape[0], out_0_shape[1])(Reshape((latent_dim3, 1))(z_mean))
# out_0_shape = y label값의 형태만큼, predicted label값을 regression으로 만들어낼거임.

D1 = Dense(latent_dim2, activation='relu')  # into 500
D2 = Dense(latent_dim1, activation='relu')  # into 2048
D3 = Dense(n_feat, activation='sigmoid')  # into 2400
h_decoded1 = D1(z)  # latent space에서 샘플링한 z를 인풋으로하여 아웃풋도 latent space인 fullyconnected layer
h_decoded2 = D2(h_decoded1)
x_decoded_mean = D3(h_decoded2)

print('z', z.shape)
print('h_decoded1', h_decoded1.shape)
print('h_decoded2', h_decoded2.shape)
print('x_decoded_mean', x_decoded_mean.shape)
print('==============================')
out_0 = EE.networks.decoder(x_decoded_mean, shape, norm=1) # 위에서만든 layer로 디코더 실행. 근데 사실상 이 디코더에 오기까지 오리지날 트레인 x를 인코드하는거부터 시작됨. vae.


def vae_loss(img, rec):
    kl_loss = - 0.5 * KB.mean(1 + z_log_sigma - KB.square(z_mean) - KB.exp(z_log_sigma), axis=-1)
    return w_1 * kl_loss

def rec_loss(img, rec):
    mse = EE.losses.mse(img, rec)
    return mse

def pred_loss(y_true, y_pred):
    # ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    ce = EE.losses.categorical_crossentropy(y_true, y_pred)
    return (1 - w_1) * ce


loss = [rec_loss, pred_loss, vae_loss]

model_train = K.models.Model([inp_0], [out_0, out_1, out_0]) #inp_0: train data, out_0 : reconstruted img, out_1: predicted label. (vae)에서 쌓은 레이어로 모델만듦

model_rec_z = K.models.Model([inp_0], [out_0, z_mean])
model_rec_z_y = K.models.Model([inp_0], [out_0, z_mean, out_1])
model_au_int = K.models.Model([inp_0], [out_1])


sum_vac_disfa_dir = log_dir_model + '/z_val/disfa/' + args.log_dir
sum_mult_out_dir = 'res_disfa_' + str(args.warming).zfill(4) + '.csv/' + args.log_dir

if source_data != 'init':
    if args.num_au == 12 and au_index < 12:
        vae_model = VAE((160, 240, 1), batch_size, 12)
        vae_model.loadWeight(args.restored_model + '.h5', None, None)
        for i in range(len(model_train.layers) - 1):
            loaded = vae_model.model_train.layers[i].get_weights()
            model_train.layers[i].set_weights(loaded)
            if args.fine_tune:
                model_train.layers[i].trainable = False
                print(model_train.layers[i], model_train.layers[i].trainable)
                sum_mult_out_dir += '/fine_tune'
                sum_vac_disfa_dir += '/fine_tune'
        w = vae_model.model_train.get_weights()[58][:, au_index]
        b = vae_model.model_train.get_weights()[59][au_index]
        w = w.reshape(latent_dim3, 1, 2)
        b = b.reshape(1, 2)
        model_train.layers[-1].set_weights([w, b])
        print("after: ", model_train.layers[-1].get_weights())
    else:
        model_train.load_weights(args.restored_model + '.h5')
        if args.fine_tune:
            for i in range(len(model_train.layers) - 1):
                model_train.layers[i].trainable = False
                print(model_train.layers[i], model_train.layers[i].trainable)
                sum_mult_out_dir += '/fine_tune'
                sum_vac_disfa_dir += '/fine_tune'

    print(">>>>>>>>> model loaded from ", args.restored_model)
    print(model_train.layers[len(model_train.layers) - 1], model_train.layers[len(model_train.layers) - 1].trainable)

if args.decoder is False:
    for layer in model_train.layers[20:35]:
        layer.trainable = False
    for layer in model_train.layers:
        print(layer, layer.trainable)


if not os.path.exists(sum_vac_disfa_dir):
    os.makedirs(sum_vac_disfa_dir)

model_train.compile(
    optimizer=K.optimizers.Adadelta(
        lr=args.lr,
        rho=0.95,
        epsilon=1e-08,
        decay=0.0
    ),
    loss=loss
)

model_train.summary()

# model_train.compile(K.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss=loss)
# model_train.compile(K.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004),
#                     loss=loss)
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_softmaxpdf_1_loss', patience=3, verbose=1)

model_train.fit_generator(
        generator = GEN_TR,
    samples_per_epoch=960,  # number of samples to process before going to the next epoch.
        validation_data=GEN_TE, # integer, total number of iterations on the data.
        nb_val_samples = 5000, # number of samples to use from validation generator at the end of every epoch.
    initial_epoch=args.init_epoch,
        nb_epoch = nb_iter,
        max_q_size = 4,
        callbacks=[
            # early_stopping,
            EE.callbacks.summary_multi_output(
                gen_list = (generator(TR, False, 1), generator(TE, False, 1)),
                predictor = model_au_int.predict, # predicted lable만을 예측, 이때는 augmented 되지 않은 train data를 이용하기 위해 분리?
                batch_size = batch_size,
                title = ['TR','TE'],
                one_hot=True,
                log_dir=sum_mult_out_dir,
            ),
            EE.callbacks.summary_vac_disfa(
                gen = generator(TE, False, s=True), # data augment 되지 않은, 형태가 [img], [img, lab, img]인 데이터
                predictor = model_rec_z_y.predict, # reconstructed x와 z mean얻어냄
                log_dir=sum_vac_disfa_dir,
                nb_batches=10,
                batch_size=batch_size,
                ),
            # K.callbacks.ModelCheckpoint(model_name),
            ]
        )

if nb_iter > 0: model_train.save_weights(model_name)
import cv2

if args.deep_feature is not '':
    vae_model = VAE((160, 240, 1), batch_size, args.num_au)
    vae_model.loadWeight(args.restored_model + '.h5')

    path = '/home/ml1323/project/robert_data/DISFA/detected_disfa/'
    all_subjects = os.listdir(path)

    for subject in all_subjects:
        per_sub_path = path + subject
        files = os.listdir(per_sub_path)
        detected_frame_idx = [int(elt.split('frame')[1].split('_')[0]) for elt in files]
        detected_frame_idx = list(set(detected_frame_idx))

        imgs = [cv2.imread(per_sub_path + "/frame" + str(i) + "_0.jpg") for i in detected_frame_idx]
        pre_processed_img_arr = []
        for img in imgs:
            img2, _, _ = pp.transform(img, preprocessing=True, augmentation=False)
            pre_processed_img_arr.append(img2)
        pre_processed_img_arr = np.array(pre_processed_img_arr)
        print('pre_processed_img_arr:', pre_processed_img_arr.shape)
        model_deep_feature = vae_model.model_deep_feature.predict(pre_processed_img_arr)
        print('len deep feat:', len(model_deep_feature))
        print('len files:', len(detected_frame_idx))
        if not os.path.exists(args.deep_feature):
            os.makedirs(args.deep_feature)
        save_path = args.deep_feature + '/' + subject + '.csv'
        with open(save_path, 'a') as f:
            for i in range(len(model_deep_feature)):
                out_csv = np.hstack((subject, "frame" + detected_frame_idx[i], [str(x) for x in model_deep_feature[i]]))
                f.write(','.join(out_csv) + '\n')
        print(">>>>>>>>done: ", subject, len(model_deep_feature))


end_time = datetime.now()
elapse = end_time - start_time
print("=======================================================")
print(">>>>>> elapse time: " + str(elapse))
print("=======================================================")
