import pickle
from EmoEstimator.utils.evaluate import print_summary
import numpy as np
import sys

r_path = '/home/ml1323/project/robert_code/new/disfa/seed0/m1_ce_0.01co_shuffle1_adadelta/cls_2.mbs_14.ubs_10.numstep1.updatelr0.01.metalr0.01/adaptation/update_lr0.008.metalr0.008.lambda0.01.num_updates1.meta_iter'
kshot = sys.argv[1]
max_seed = sys.argv[2]
iter = sys.argv[3]

all_seed_info = []


for seed in range(int(max_seed)):
    print("")
    print("============================================================")
    print("seed " + str(seed))
    print("============================================================")
    print("")
    path = r_path + iter + '/' + kshot + 'kshot/seed' + str(seed)
    y_lab_all = []
    y_hat_all = []
    f1_scores_per_seed = []
    for subject_idx in range(13):
        print("=============================subject_idx: ", subject_idx)
        file = pickle.load(open(path + '/predicted_subject' + str(subject_idx) + '.pkl', 'rb'), encoding='latin1')
        y_lab = file['y_lab']
        y_hat = file['y_hat']
        y_lab_all.append(y_lab)
        y_hat_all.append(y_hat)
        out = print_summary(y_hat, y_lab, log_dir="./logs/result/" + "/test.txt")
        f1_score = list(out['data'][5])
        # add avg throughout all AUs as the last elt
        f1_score.append(np.average(f1_score))
        # stack each subject's f1-score
        f1_scores_per_seed.append(f1_score)
        print("-- num of samples:", len(file['used_samples']))

    print(">> y_lab_all shape:", np.vstack(y_lab_all).shape)
    print(">> y_hat_all shape:", np.vstack(y_hat_all).shape)

    print('-------------------- avg --------------------')
    averaged_f1 = np.average(f1_scores_per_seed, axis=0)
    print(averaged_f1)
    print('---------------- concatenated ---------------')
    out = print_summary(np.vstack(y_hat_all), np.vstack(y_lab_all), log_dir="./logs/result/" + "/test.txt")
    long_f1 = list(out['data'][5])
    # add avg throughout all AUs as the last elt
    long_f1.append(np.average(long_f1))
    print(long_f1)
    f1_scores_per_seed.append(averaged_f1)
    f1_scores_per_seed.append(long_f1)
    all_seed_info.append(f1_scores_per_seed)

std_avg = np.std([elt[-2] for elt in all_seed_info])
std_long = np.std([elt[-1] for elt in all_seed_info])

mean_avg = np.mean([elt[-2] for elt in all_seed_info])
mean_long = np.mean([elt[-1] for elt in all_seed_info])

print("=======================================")
print('mean_avg: ', np.round(mean_avg, 2))
print('std_avg: ', np.round(std_avg, 2))
print("---------------------------------------")
print('mean_long: ', np.round(mean_long, 2))
print('std_long: ', np.round(std_long, 2))
print("=======================================")
