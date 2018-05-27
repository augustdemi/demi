import numpy as np
import os
import h5py
from sklearn.model_selection import train_test_split


############### img ############
path = "/home/ml1323/project/robert_data/DISFA/h5_3/"
# path = "D:/연구/프로젝트/Robert Walecki - deep_coder_submission_code/h5/"

files = os.listdir(path)
imgs = []
labels = []
subjects = []
for file in files:
    print(">>>> file: " + file)
    hf = h5py.File(path+file, 'r')
    img=hf['img'].value[0:4844]
    label=hf['lab'].value[0:4844]
    subject_number = int(file.split(".")[0].split("SN")[1])
    n_data = len(img)
    print(len(img))
    print(len(label))
    if(len(imgs) ==0):
        imgs = img
        labels = label
        subjects = np.ones(n_data) * subject_number
    else:
        imgs=np.vstack((imgs, img))
        labels=np.vstack((labels, label))
        subjects = np.vstack((subjects, np.ones(n_data) * subject_number))
    hf.close()

print(subjects.shape)
total_n = len(imgs)
subjects=subjects.reshape(total_n,)

half_img, dump_img, half_label, dump_label, half_sub, dump_sub = train_test_split(imgs, labels, subjects, test_size=0.5)
temp_img, test_img, temp_label, test_label, temp_sub, test_sub = train_test_split(half_img, half_label, half_sub, test_size=0.1)
train_img, validate_img, train_label, validate_label, train_sub, validate_sub= train_test_split(temp_img, temp_label, temp_sub, test_size=0.1)
print(len(train_img))
print(len(test_img))
print(len(validate_img))
print(train_sub.reshape(len(train_sub),).shape)
# save_path = "D:/연구/프로젝트/Robert Walecki - deep_coder_submission_code/split/"
save_path = "/home/ml1323/project/robert_data/DISFA/split/"
hf = h5py.File(save_path+"train.h5", 'w')
hf.create_dataset('img', data=train_img)
hf.create_dataset('lab', data=train_label)
hf.create_dataset('sub', data=train_sub.reshape(len(train_sub),))
hf.close()
hf = h5py.File(save_path+"test.h5", 'w')
hf.create_dataset('img', data=test_img)
hf.create_dataset('lab', data=test_label)
hf.create_dataset('sub', data=test_sub.reshape(len(test_sub),))

hf.close()
hf = h5py.File(save_path+"validate.h5", 'w')
hf.create_dataset('img', data=validate_img)
hf.create_dataset('lab', data=validate_label)
hf.create_dataset('sub', data= validate_sub.reshape(len(validate_sub),))

hf.close()

