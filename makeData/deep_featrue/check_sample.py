import pickle
import os

path = '/home/ml1323/project/robert_code/new/check_labels/'
files = os.listdir(path)

test_subjects = ['SN017', 'SN018', 'SN021', 'SN023', 'SN024', 'SN025', 'SN026', 'SN027', 'SN028', 'SN029', 'SN030',
                 'SN031', 'SN032']

num_frame = {}

for subject in test_subjects:
    one_subj_files = [f for f in files if f.startswith(subject)]
    frames = []
    for file in one_subj_files:
        contents = pickle.load(open(path + file, 'rb'), encoding='latin1')
        frames.extend(contents['off'])
        frames.extend(contents['on'])
    frames = set(frames)
    num_frame.update({subject: len(frames)})
print(num_frame)
