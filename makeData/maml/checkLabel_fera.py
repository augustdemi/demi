import os
import numpy as np

lable_dir = '/home/ml1323/project/robert_data/FERA/BP4D-AUCoding/AUCoding/'
au_idx = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23]

label_file_list = os.listdir(lable_dir)
subjects = set([file_name.split('_')[0] for file_name in label_file_list])  # distinct subjects 구해서

missing_data_all = {}
num_pos_all = {}

for subject in subjects:
    label_files = [file_name for file_name in label_file_list if
                   file_name.startswith(subject)]  # subject으로 시작하는 파일 8개만 모아서
    missing_data = 0
    num_pos = [0] * len(au_idx)
    for file_name in label_files:
        f = open(lable_dir + file_name)
        f.readline()
        while True:
            line = f.readline().split(',')
            if len(line) == 1: break
            frame = line[0]
            codes = np.array(line)[au_idx]
            codes = [int(code) for code in codes]
            if 9 in codes:
                missing_data += 1
            else:
                for i in range(len(codes)):
                    if codes[i] > 0: num_pos[i] += 1
    missing_data_all.update({subject: missing_data})
    num_pos_all.update({subject: num_pos})

print(missing_data_all)
print(num_pos_all)
