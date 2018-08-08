import h5py
import pickle

f = h5py.File("/home/ml1323/project/robert_data/DISFA/h5_maml/train.h5")
save_path = "/home/ml1323/project/robert_data/DISFA/h5_maml/train_label.pkl"

lab = f['lab']
sub = f['sub']
bb = set(sub.value)
lab_shape = lab.shape
subs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16]
subs = [17, 18, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

total = []
on_label_per_au = {}
# for one au,
for i in range(lab_shape[1]):
    total.append(lab_shape[0])
    on_label_cnt_per_sub = []
    # for one subject,
    for j in subs:
        # find the index of data of which subject is j
        sub_index = [idx for idx in range(sub.shape[0]) if sub.value[idx] == j]
        cnt = 0
        for k in sub_index:
            cnt += lab[k, i].argmax()
        on_label_cnt_per_sub.append(cnt)
    on_label_per_au.update({i: on_label_cnt_per_sub})

out = open(save_path, 'wb')
pickle.dump(on_label_per_au, out, protocol=2)
out.close()

print(total)  # 67788
# print(on_label_cnt) # [4621, 3579, 9742, 1171, 7916, 3423, 15288, 4957, 5891, 2065, 26313, 15481]
