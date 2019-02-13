import pickle
from EmoEstimator.utils.evaluate import print_summary
import numpy as np
import sys

path = '/home/ml1323/project/robert_code/new/disfa/seed0/m1_ce_0.01co_shuffle1_adadelta/cls_2.mbs_14.ubs_10.numstep1.updatelr0.01.metalr0.01/'
path += sys.argv[1]
y_lab_all = []
y_hat_all = []
f1_scores = []
for subject_idx in range(12):
    file = pickle.load(open(path + '/predicted_subject' + str(subject_idx) + '.pkl', 'rb'), encoding='latin1')
    y_lab = file['y_lab']
    y_hat = file['y_hat']
    y_lab_all.append(y_lab)
    y_hat_all.append(y_hat)
    out = print_summary(y_hat, y_lab, log_dir="./logs/result/" + "/test.txt")
    f1_scores.append(out['data'][5])

print(">> y_lab_all shape:", np.vstack(y_lab_all).shape)
print(">> y_hat_all shape:", np.vstack(y_hat_all).shape)
print('-------------------- avg --------------------')
print(np.average(f1_scores, axis=0))
print('---------------- concatenated ---------------')
print_summary(np.vstack(y_hat_all), np.vstack(y_lab_all), log_dir="./logs/result/" + "/test.txt")
