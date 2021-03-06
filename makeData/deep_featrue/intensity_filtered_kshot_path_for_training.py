import random
import os

original_frame_path = "/home/ml1323/project/robert_data/DISFA/detected_disfa/"
save_path = "/home/ml1323/project/robert_data/DISFA/kshot_path_filter_8au/"
all_au = ['au1', 'au2', 'au4', 'au6', 'au9', 'au12', 'au25', 'au26']
cnt = 0
cnta_arr = []
for subject in os.listdir(original_frame_path):
    detected_img_files = os.listdir(original_frame_path + subject)
    detected_frame_idx = [int(elt.split('frame')[1].split('_')[0]) for elt in detected_img_files]
    detected_frame_idx = list(set(detected_frame_idx))
    detected_frame_idx.sort()
    all_intensities = [0] * (detected_frame_idx[-1] + 1)
    #### filter ####
    for au in all_au:
        label_path = "/home/ml1323/project/robert_data/DISFA/label/" + subject + "/" + subject + "_" + au + ".txt"
        f = open(label_path, 'r')
        all_labels = f.readlines()
        # print(all_labels)

        for idx in detected_frame_idx:
            # print(int(all_labels[idx].split(",")[1].split("\n")[0]))
            try:
                intensity = int(all_labels[idx].split(",")[1].split("\n")[0])
                all_intensities[idx] += intensity
            except:
                # print("out of index: ", idx, " for au, subject: ", au, subject)
                continue
    detected_frame_idx = [idx for idx in range(len(all_intensities)) if all_intensities[idx] > 6]
    cnt += len(detected_frame_idx)
    cnta_arr.append(len(detected_frame_idx))

    # print('test_a len: ', len(test_a_idx))
    # print('test_b len: ', len(test_b_idx))

    ########### 위의 각 subject 별 fixed test_a, test_b에 대해서, 각 au별로 on/off를 구해 분리해둠.
    ########### 그리곤 test_b셋에대해서 testset pickle값을 만들어 둬서 이 전체 데이터 셋에 대해서 evaluation할 것임.
    summary_test_a_on = []
    summary_test_a_off = []
    for au in all_au:
        label_path = "/home/ml1323/project/robert_data/DISFA/label/" + subject + "/" + subject + "_" + au + ".txt"
        f = open(label_path, 'r')
        all_labels = f.readlines()

        test_a_on_idx = []
        test_a_off_idx = []
        for idx in detected_frame_idx:
            try:
                intensity = int(all_labels[idx].split(",")[1].split("\n")[0])
            except:
                # print("test_a_on_idx - out of index: ", idx, " for au, subject: ", au, subject)
                continue
            if intensity > 0:
                test_a_on_idx.append(idx)
            else:
                test_a_off_idx.append(idx)

        summary_test_a_on.append(len(test_a_on_idx))
        summary_test_a_off.append(len(test_a_off_idx))
        save_path_teat_a_per_au_sub = save_path + "/train/" + au + "/" + subject

        if not os.path.exists(save_path_teat_a_per_au_sub + "/on"): os.makedirs(save_path_teat_a_per_au_sub + "/on")
        if not os.path.exists(save_path_teat_a_per_au_sub + "/off"): os.makedirs(save_path_teat_a_per_au_sub + "/off")

        # copy on intensity frames for train
        file_path_to_save = []
        with open(save_path_teat_a_per_au_sub + "/on/file_path.csv", 'w') as f:
            for i in test_a_on_idx:
                file_path_to_save.append(original_frame_path + subject + "/frame" + str(i) + "_0.jpg")
            f.write(','.join(file_path_to_save))

        # copy off intensity frames for train
        file_path_to_save = []
        with open(save_path_teat_a_per_au_sub + "/off/file_path.csv", 'w') as f:
            for i in test_a_off_idx:
                file_path_to_save.append(original_frame_path + subject + "/frame" + str(i) + "_0.jpg")
            f.write(','.join(file_path_to_save))
        print(">>>>> done: ", au)

    print(summary_test_a_on)
    print(summary_test_a_off)
    print("========================================= done: ", subject)
print('cnt', cnt)
print('cnta_arr', cnta_arr)
