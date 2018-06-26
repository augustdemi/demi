import pickle
import random
import os
import numpy as np

test_num= 200
test_file_pkl = "test_test"
path = "/home/ml1323/project/robert_data/DISFA/kshot_rest/rest/"
subjects = os.listdir(path)
subjects.sort()
subject_folders = [os.path.join(path, subject) for subject in subjects][14:27]
test_file_names = []
y_lab = []
for subject_folder in subject_folders:
    off_files = random.sample(os.listdir(subject_folder + "/off"), test_num)
    test_file_names.extend([os.path.join(subject_folder + "/off", off_file) for off_file in off_files])
    on_files = random.sample(os.listdir(subject_folder + "/on"), test_num)
    test_file_names.extend([os.path.join(subject_folder + "/on", on_file) for on_file in on_files])
    lab = np.zeros((test_num, 2))
    lab[:, 0] = 1
    y_lab.extend(lab)
    lab = np.zeros((test_num, 2))
    lab[:, 1] = 1
    y_lab.extend(lab)
np.random.seed(3)
np.random.shuffle(test_file_names)
y_lab = np.array(y_lab)
y_lab = y_lab.reshape(y_lab.shape[0], 1, 2)
np.random.seed(3)
np.random.shuffle(y_lab)


save_path = "/home/ml1323/project/robert_data/DISFA/kshot_rest/testset/" + test_file_pkl + ".pkl"
out = open(save_path, 'wb')
pickle.dump({'test_file_names': test_file_names, 'y_lab': y_lab}, out, protocol=2)

