import h5py
import random
import numpy as np
from tqdm import tqdm
import os

def merge_h5(in_path_list, out_path, subject_ids=None, shuffle=False, downsample=1):

    if subject_ids:
        sub_idx = []

    if os.path.exists(out_path+'.tmp.h5'):
        os.remove(out_path+'.tmp.h5')

    if os.path.exists(out_path):
        os.remove(out_path)

    f_tmp = h5py.File(out_path+'.tmp.h5')
    f_out = h5py.File(out_path)

    for i in tqdm(range(len(in_path_list))):
        f_in = in_path_list[i]

        with h5py.File(f_in) as f:

            if i==0:
                for group in f.keys():
                    dat = f[group][::downsample]

                    f_tmp.create_dataset(
                            group,
                            data=dat,
                            maxshape=(None,*f[group].shape[1:])
                            )

            else:
                for group in f.keys():

                    dat = f[group][::downsample]

                    f_tmp[group].resize((
                        f_tmp[group].shape[0] + dat.shape[0] , *f[group].shape[1:]
                        ))
                    f_tmp[group][-dat.shape[0]:] = dat

        if subject_ids:
            sub = subject_ids[i]
            nb_samples = dat.shape[0]
            sub_idx.extend([sub]*nb_samples)

    if subject_ids:
        f_tmp.create_dataset('sub',data=np.array(sub_idx))



    idx = np.arange(f_tmp[list(f_tmp.keys())[0]].shape[0])
    if shuffle:random.shuffle(idx)

    for i in tqdm(range(len(idx))):
        frame_num = idx[i]

        if i==0:
            for group in f_tmp.keys():
                try:
                    dat = f_tmp[group][frame_num][None,::]
                except IndexError:
                    dat = f_tmp[group][frame_num][None]

                f_out.create_dataset(
                        group,
                        data=dat,
                        maxshape=(None,*f_tmp[group].shape[1:])
                        )
        else:
            for group in f_out.keys():
                f_out[group].resize((
                    f_out[group].shape[0]+1, *f_out[group].shape[1:]
                    ))
                f_out[group][-1]=f_tmp[group][frame_num]


    f_out.close()
    f_tmp.close()
    os.remove(out_path+'.tmp.h5')
