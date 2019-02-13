import pickle
import os

path = '/home/ml1323/project/robert_code/new/check_labels/'
files = os.listdir(path)

test_subjects = ['SN017', 'SN018', 'SN021', 'SN023', 'SN024', 'SN025', 'SN026', 'SN027', 'SN028', 'SN029', 'SN030',
                 'SN031', 'SN032']

num_frame = {}

for subject_idx in range(13, 13):
    print('=================== subject_idx: ', subject_idx)
    one_subj_files = [f for f in files if f.startswith(str(subject_idx) + '_')]
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
    num_frame.update({subject_idx: len(frames)})
print(num_frame)
