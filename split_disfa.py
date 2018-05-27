
import numpy as np
import os
import h5py
from sklearn.model_selection import train_test_split


############### img ############
path = "/home/ml1323/project/robert_data/DISFA/h5/"
# path = "D:/연구/프로젝트/Robert Walecki - deep_coder_submission_code/h5/"

files = os.listdir(path)
imgs = []
labels = []
subjects = []
for file in files:
    print(">>>> file: " + file)
    hf = h5py.File(path+file, 'r')
    img=hf['img'].value[0:4845]
    label=hf['lab'].value[0:4845]
    subject_number = int(file.split(".")[0].split("SN")[1])
    half_img, dump_img, half_label, dump_label = train_test_split(img, label, test_size=0.5)

    if(len(imgs) ==0):
        imgs = half_img
        labels = half_label
        subjects = np.ones(len(half_img)) * subject_number
    else:
        imgs=np.vstack((imgs, half_img))
        labels=np.vstack((labels, half_label))
        subjects = np.vstack((subjects, np.ones(len(half_img)) * subject_number))
    print(len(half_img))
    hf.close()




temp_img, test_img, temp_label, test_label, temp_sub, test_sub = train_test_split(imgs, labels, subjects, test_size=0.1)
train_img, validate_img, train_label, validate_label, train_sub, validate_sub= train_test_split(temp_img, temp_label, temp_sub, test_size=0.1)
print(len(train_img))
print(len(test_img))
print(len(validate_img))

# save_path = "D:/연구\프로젝트/Robert Walecki - deep_coder_submission_code/split/"
save_path = "/home/ml1323/project/robert_data/DISFA/split/"
hf = h5py.File(save_path+"train.h5", 'w')
hf.create_dataset('img', data=train_img)
hf.create_dataset('lab', data=train_label)
hf.create_dataset('sub', data=train_sub)
hf.close()
hf = h5py.File(save_path+"test.h5", 'w')
hf.create_dataset('img', data=test_img)
hf.create_dataset('lab', data=test_label)
hf.create_dataset('sub', data=test_sub)

hf.close()
hf = h5py.File(save_path+"validate.h5", 'w')
hf.create_dataset('img', data=validate_img)
hf.create_dataset('lab', data=validate_label)
hf.create_dataset('sub', data= validate_sub)

hf.close()

