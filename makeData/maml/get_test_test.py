import pickle
import os
import numpy as np

au = "au25"
path = "/home/ml1323/project/robert_data/DISFA/kshot_rest/" + au
save_path = path + "/testset/"
if not os.path.exists(save_path): os.makedirs(save_path)

path += "/rest"
subjects = os.listdir(path)
subjects.sort()

for subject in subjects[14:27]:
    subject_folder = os.path.join(path, subject)
    test_file_names = []
    y_lab = []
    off_files = os.listdir(subject_folder + "/off")
    test_file_names.extend([os.path.join(subject_folder + "/off", off_file) for off_file in off_files])
    on_files = os.listdir(subject_folder + "/on")
    test_file_names.extend([os.path.join(subject_folder + "/on", on_file) for on_file in on_files])

    lab = np.zeros((len(off_files), 2))
    lab[:, 0] = 1
    y_lab.extend(lab)
    lab = np.zeros((len(on_files), 2))
    lab[:, 1] = 1
    y_lab.extend(lab)

    np.random.seed(3)
    np.random.shuffle(test_file_names)
    y_lab = np.array(y_lab)
    y_lab = y_lab.reshape(y_lab.shape[0], 1, 2)
    np.random.seed(3)
    np.random.shuffle(y_lab)
    out = open(save_path + subject + ".pkl", 'wb')
    pickle.dump({'test_file_names': test_file_names, 'y_lab': y_lab}, out, protocol=2)

