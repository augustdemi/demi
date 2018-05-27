import EmoEstimator as EE
import EmoData as ED
import keras.backend as KB
from keras.layers import Dense, Lambda, Input, Reshape
import keras as K
import numpy as np
import h5py
import argparse
from datetime import datetime
start_time = datetime.now()
parser = argparse.ArgumentParser(description='extract feace images from raw datasets')
parser.add_argument("-i","--input",  type=str, default='init', help="files created from GP")
parser.add_argument("-o","--output", type=str, default='./model_output/disfa_final_init', help="files creaded from VAE")
parser.add_argument("-n","--nb_iter",type=int, default=1, help="number of VAE iterations")
parser.add_argument("-w","--warming",type=int, default=1, help="factor on kl loss")
parser.add_argument("-tr","--training_data",type=str, default='./robert_data/MNIST/trainSet', help="path to training data set")
parser.add_argument("-te","--test_data",type=str, default='./robert_data/MNIST/testSet/', help="path to test data set")
args = parser.parse_args()

# lins input
source_data = args.input
nb_iter = args.nb_iter


if source_data=='init':
    target_std_vec = np.ones(2000)
    target_mean_vec = np.zeros(2000)
else:

    with h5py.File(source_data + '_va.h5') as f:
        dat = f['mean_reconstr'][::]
        target_mean_vec= dat.mean(0)
        target_std_vec= dat.std(0)
        print(target_mean_vec.mean())
        print(target_std_vec.mean())



batch_size = 10 # dont change it!
log_dir_model = './model'
latent_dim = 2000
w_1 = args.warming / 50


#==================================================
import cv2
import os


train_p = "/home/ml1323/project/robert_data/MNIST/trainingSet/"
#train_p = "/home/ml1323/project/robert_data/resized_MNIST/trainingSet/"


train_img_data = []
train_label_data = []
train_pair = []
for i in range(0,10):
    dir = train_p + str(i)
    files = os.listdir(dir)
    for file in files:
        img = cv2.imread(dir + "/" + file)
        resized_img = cv2.resize(img, (160,240))
        train_img_data.append(resized_img)
#        label = np.zeros((10,2))
#        label[:,1] = 1
#        label[i] = [1,0]
        label = np.zeros(10)
        label[i] = 1 
        train_label_data.append(label)



from sklearn.model_selection import train_test_split
temp_img, test_img, temp_label, test_label = train_test_split(train_img_data, train_label_data, test_size=0.1)
train_img, validate_img,  train_label, validate_label= train_test_split(temp_img, temp_label, test_size=0.1)


train_data = {'img' : np.array(train_img), 'lab': np.array(train_label)}
val_data = {'img' : np.array(validate_img), 'lab': np.array(validate_label)}
test_data = {'img' : np.array(test_img), 'lab': np.array(test_label)}



#==================================================




TR = ED.provider.flow_from_hdf5(train_data, batch_size, padding='same')
TE = ED.provider.flow_from_hdf5(test_data, batch_size, padding='same')


pp = ED.image_pipeline.FACE_pipeline(
        histogram_normalization=True,
        grayscale=True,
        rotation_range = 3,
        width_shift_range = 0.03,
        height_shift_range = 0.03,
        zoom_range = 0.03,
        random_flip = True,
        )


def generator(dat_dict, aug, mod=0):
    while True:

        img = next(dat_dict['img'])
        lab = next(dat_dict['lab'])
        img, pts, pts_raw = pp.batch_transform(
                img,
                preprocessing=True,
                augmentation=aug)
#        lab = lab.argmax(2)
        if mod==1:
            yield [img], [lab]
        else:
            yield [img], [img, lab, img]


GEN_TR = generator(TR, True) # 
GEN_TE = generator(TE, False)

# X,Y,C dimenstion # ========================== start ==========================
X, Y = next(GEN_TR) #
inp_0_shape = X[0].shape[1:]
out_0_shape = Y[1].shape[1:]

print(inp_0_shape)
print(out_0_shape)
inp_0       = Input(shape=inp_0_shape) # 
emb, shape  = EE.networks.encoder(inp_0, norm=1) # con
from numpy import prod
from keras.layers import Dropout
n_feat = prod(shape)

emb = Dropout(0.5)(emb)
z_mean      = Dense(latent_dim)(emb) #
z_log_sigma = Dense(latent_dim)(emb) #

def sampling(args): 
    z_mean, z_log_sigma = args
    epsilon = []
    for m, s in zip(target_mean_vec, target_std_vec):
        epsilon.append(KB.random_normal(shape=[batch_size, 1], mean=m, std=s))
    epsilon = KB.concatenate(epsilon, 1)
    return z_mean + KB.exp(z_log_sigma) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma]) 
out_1 = EE.layers.logREG(out_0_shape[0])(z_mean)

D1 = Dense(latent_dim, activation='relu')
D2 = Dense(n_feat, activation='sigmoid')  
h_decoded = D1(z) 
print(n_feat)
x_decoded_mean = D2(h_decoded)
print(x_decoded_mean)

out_0 = EE.networks.decoder(x_decoded_mean, shape, norm=1) # 


from keras import objectives
def vae_loss(img, rec):
    kl_loss = - 0.5 * KB.mean(1 + z_log_sigma - KB.square(z_mean) - KB.exp(z_log_sigma), axis=-1)
    return w_1*kl_loss

def rec_loss(img, rec):
    mse = EE.losses.mse(img, rec)
    return mse

def pred_loss(img, rec):
    mse = EE.losses.mse(img, rec)
    return (1-w_1)*mse

loss  = [rec_loss, pred_loss, vae_loss]

model_train = K.models.Model([inp_0], [out_0, out_1, out_0]) 

model_rec_z = K.models.Model([inp_0], [out_0, z_mean])
model_rec_z_y = K.models.Model([inp_0], [out_0, z_mean, out_1])
model_au_int= K.models.Model([inp_0], [out_1]) #??????????????????

inp_1 = Input(shape=[2000])
h1 = D1(inp_1)
x1 = D2(h1) 
out_1  = EE.networks.decoder(x1, shape, norm=1) 


rec = K.models.Model(inp_1, out_1)
if source_data!='init':
    rec.load_weights('./model_vae/model.h5', by_name=True) # ========================== weight ==========================


model_train.compile(
        optimizer = K.optimizers.Adadelta(
            lr = .1,
            rho = 0.95,
            epsilon = 1e-08,
            decay = 0.0
            ),
        loss = loss
        )

model_train.fit_generator( 
        generator = GEN_TR,
        samples_per_epoch = 1000,
        validation_data=GEN_TE,
        nb_val_samples = 5000,
        nb_epoch = nb_iter,
        max_q_size = 4,
        callbacks=[
            EE.callbacks.summary_multi_output(
                gen_list = (generator(TR, False, 1), generator(TE, False, 1)),
                predictor = model_au_int.predict,
                nb_batches=1,
                batch_size = batch_size,
                title = ['TR','TE'],
                log_dir = 'res_'+str(args.warming).zfill(4)+'.csv',
            ),
            EE.callbacks.summary_vac(
                gen = GEN_TE, 
                predictor = model_rec_z_y.predict, 
                log_dir = log_dir_model,
                nb_batches=100,
                batch_size=batch_size,
                ),
            K.callbacks.ModelCheckpoint('./model_vae/model.h5'),
            ]
        )

end_time = datetime.now()
elapse = end_time - start_time
print("=======================================================")
print(">>>>>> elapse time: " + str(elapse))
print("=======================================================")
# import numpy as np
# for dat, dset in zip(['./robert_data/DATA_2/disfa_va.h5','./robert_data/DATA_2/disfa_te.h5'],['_va','_te']):
#
#     TR = ED.provider.flow_from_hdf5(val_data, batch_size, padding='same')
#     def generator(dat_dict):
#         while True:
#             img = next(dat_dict['img'])
#             lab = next(dat_dict['lab'])
#             img, pts, pts_raw = pp.batch_transform(
#                     img,
#                     preprocessing=True,
#                     augmentation=False)
#             yield img, lab
#
#     GEN_TR = generator(TR)
#
#     X_out, Y_out, Z_out = [], [], []
#     for i in range(500):
#         img, lab = next(GEN_TR)
#         rec, feat, pred = model_rec_z_y.predict(img) # out_0 : reconstruted img, z_mean, out_1: predicted label
#         X_out.append(feat)
#         Y_out.append(lab)
#         Z_out.append(pred)
#
#     X_out = np.vstack(X_out)
#     Y_out = np.vstack(Y_out)
#     Z_out = np.vstack(Z_out)
#     print(Y_out.shape)
#
#     with h5py.File(args.output+dset+'.h5') as f:
#         f.create_dataset('lab',data=Y_out)
#         f.create_dataset('feat',data=X_out)
#         f.create_dataset('pred',data=Z_out)
