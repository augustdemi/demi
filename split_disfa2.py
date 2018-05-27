import numpy as np
import os
import h5py
from sklearn.model_selection import train_test_split


############### img ############
path = "/gpu/homedirs/ml1323/project/robert_data/DISFA/h5/"

files = os.listdir(path)
imgs = []
labels = []
for file in files:
    print(">>>> file: " + file)
    hf = h5py.File(path+file, 'r')
    img=hf['img'].value.tolist()[0:4845]
    label=hf['lab'].value.tolist()[0:4845]
    half_img, dump_img, half_label, dump_label = train_test_split(img, label, test_size=0.5)

    imgs.extend(half_img)
    labels.extend(half_label)
    print(len(imgs))
    print(len(labels))
    hf.close()


temp_img, test_img, temp_label, test_label = train_test_split(imgs, labels, test_size=0.1)
train_img, validate_img, train_label, validate_label= train_test_split(temp_img, temp_label, test_size=0.1)
print(len(train_img))
print(len(test_img))
print(len(validate_img))
print(len(validate_label))


save_path = "/gpu/homedirs/ml1323/project/robert_data/DISFA/split/"
hf = h5py.File(save_path+"train.h5", 'w')
hf.create_dataset('img', data=np.array(train_img))
hf.create_dataset('lab', data=np.array(train_label))
hf.close()
hf = h5py.File(save_path+"test.h5", 'w')
hf.create_dataset('img', data=np.array(test_img))
hf.create_dataset('lab', data=np.array(test_label))
hf.close()
hf = h5py.File(save_path+"validate.h5", 'w')
hf.create_dataset('img', data=np.array(validate_img))
hf.create_dataset('lab', data=np.array(validate_label))
hf.close()

