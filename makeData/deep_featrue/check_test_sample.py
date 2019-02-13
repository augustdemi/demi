import pickle
import os

path = '/home/ml1323/project/robert_code/new/check_labels/test/5shot/'
files = os.listdir(path)

test_subjects = ['SN017', 'SN018', 'SN021', 'SN023', 'SN024', 'SN025', 'SN026', 'SN027', 'SN028', 'SN029', 'SN030',
                 'SN031', 'SN032']

num_frame = {}

for subject in test_subjects:
    print('=================== subject_idx: ', subject)
    one_subj_files = [f for f in files if f.startswith(subject)]
    frames = []
    for file in one_subj_files:
        print('-----------------------------')
        print('file name: ', file)
        contents = pickle.load(open(path + file, 'rb'), encoding='latin1')
        print('off: ', contents['off'])
        print('on: ', contents['on'])
        frames.extend(contents['off'])
        frames.extend(contents['on'])
    frames = set(frames)
    num_frame.update({subject: len(frames)})
    # with open(path + 'statistics.csv', 'w') as f:
    #     for i in num_frame:
    #         out_csv = np.hstack(
    #             (subject, "frame" + str(detected_frame_idx[i]), [str(x) for x in deep_feature[i]]))
    #         f.write(','.join(out_csv) + '\n')
    # print(">>>>>>>>done: ", subject, len(deep_feature))
print(num_frame)
