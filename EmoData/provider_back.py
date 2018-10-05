import h5py
import math
import numpy as np
import threading
from glob import glob
import os.path
from skimage.io import imread

def flow_from_hdf5(
        path_to_file, 
        batch_size=64,
        padding=None,
        au_idx=12
        ):
    '''
    '''
    print(path_to_file)
    f = h5py.File(path_to_file, 'r')
    lock = threading.Lock()

    # get the sice of the first group in the hdf5 file
    data = f[[i for i in f.keys()][0]]
    nb_samples = data.shape[0]
    nb_batches = math.ceil(nb_samples/batch_size)

    all_indices = []
    if au_idx < 12:
        lab = f['lab'][:, au_idx]
        N_total_label = lab.shape[0]

        sub = f['sub']
        subject_set = list(set(sub))
        subject_set.sort()

        per_subject_on_cnt = {}
        per_subject_on_idx = {}
        per_subject_off_idx = {}
        for i in subject_set:
            per_subject_on_cnt[i] = 0
            per_subject_on_idx[i] = []
            per_subject_off_idx[i] = []

        for i in range(N_total_label):
            if lab[i][1] == 0:
                per_subject_off_idx[sub[i]].append(i)
            else:
                per_subject_on_idx[sub[i]].append(i)
                per_subject_on_cnt[sub[i]] += 1

        avg_num_on_intensity = int(np.average(list(per_subject_on_cnt.values())))
        print('>>>>>>>>>>>>>> avg_num_on_intensity', avg_num_on_intensity)

        for i in subject_set:
            final_num_on_int = min(per_subject_on_cnt[i], avg_num_on_intensity)
            required_per_subject_off_cnt = 2 * avg_num_on_intensity - final_num_on_int
            all_indices.extend(per_subject_on_idx[i][:int(final_num_on_int)])
            all_indices.extend(per_subject_off_idx[i][:int(required_per_subject_off_cnt)])
        all_indices.sort()
        nb_samples = len(all_indices)
        nb_batches = math.ceil(nb_samples / batch_size)

    print('-----------------------------------')
    print('nb_samples: ', nb_samples)
    print('nb_batches: ', nb_batches)
    print('-----------------------------------')

    def _make_generator(data):

        t0, t1  = 0, batch_size

        while True:

            t1 = min( nb_samples, t1 ) # 배치 사이즈와 샘플 갯수중 작은게 t1
            if t0 >= nb_samples: # 샘플 갯수보다 to가 크면 변수 초기화
                t0, t1 = 0, batch_size

            batch = data[t0:t1] # 이번에 돌릴 배치값 배정 : 데이터의 t0부터 t1까지
            if padding!=None and batch.shape[0]<batch_size: # 패딩 설정
                if padding=='same':
                    batch = data[-batch_size:]
                else:
                    tmp = padding*np.ones([batch_size,*batch.shape[1:]])
                    tmp[:batch.shape[0]]=batch
                    batch = tmp

            t0 += batch_size # 작업한 배치 사이즈만큼 t0와 t1늘림
            t1 += batch_size

            yield batch # 패딩이 적용된, 배치사이즈 만큼의 배치데이터 생성

    res_gen = {}
    res_gen['nb_samples']=nb_samples
    res_gen['nb_batches']=nb_batches

    if au_idx < 12:
        for key in f:
            res_gen[key] = _make_generator(f[key][all_indices])
    else:
        for key in f:
            res_gen[key] = _make_generator(f[key])
    return res_gen

def flow_from_np_array(
        X, y, 
        batch_size=64,
        padding = None 
        ):
    '''
    '''
    # get the sice of the first group in the hdf5 file
    nb_samples = X.shape[0]
    nb_batches = math.ceil(nb_samples/batch_size)

    def _make_generator(data):

        t0, t1  = 0, batch_size

        while True:

            t1 = min( nb_samples, t1 )
            if t0 >= nb_samples:
                t0, t1 = 0, batch_size

            batch = data[t0:t1]
            if padding!=None and batch.shape[0]<batch_size:
                if padding=='same':
                    batch = data[-batch_size:]
                else:
                    tmp = padding*np.ones([batch_size,*batch.shape[1:]])
                    tmp[:batch.shape[0]]=batch
                    batch = tmp

            t0 += batch_size
            t1 += batch_size

            yield batch

    res_gen = {}
    res_gen['nb_samples']=nb_samples
    res_gen['nb_batches']=nb_batches
    res_gen['img']=_make_generator(X)
    res_gen['lab']=_make_generator(y)
    return res_gen

def flow_from_folder(path_to_folder,
    batch_size=64,
    padding = None,
    sort = True 
    ):

    # load recursicely all images in given folder and subfolder
    all_img = glob(path_to_folder+'/**/*.jpg', recursive=True)
    if sort:
        all_img.sort()

    # use only that images that come with label files (txt of csv)
    valid_img = []
    valid_lab = []
    for i in all_img:
        if os.path.isfile(i[:-3]+'txt') :
            valid_img.append(i)
            valid_lab.append(i[:-3]+'txt')
        if os.path.isfile(i[:-3]+'csv'):
            valid_img.append(i)
            valid_lab.append(i[:-3]+'csv')

    nb_samples = len(valid_img)
    nb_batches = math.ceil(nb_samples/batch_size)




    def _make_generator(data):

        t0, t1  = 0, batch_size

        while True:

            t1 = min( nb_samples, t1 )
            if t0 >= nb_samples:
                t0, t1 = 0, batch_size

            if data[0][-3:]=='txt' or data[0][-3:]=='csv':
                batch = []
                for fname in data[t0:t1]:
                    with open(fname) as f:
                            content = f.readlines()
                            content = [x.strip() for x in content] 
                            content = np.array([np.float32(x.split(',')[1:]) for x in content])
                            batch.append(content)
            else:
                batch = [np.float32(imread(i)) for i in data[t0:t1]]


            while padding!=None and len(batch)<batch_size:
                if padding=='same':
                    batch.append(batch[-1])
                else:
                    batch.append(padding*np.ones_like(batch[-1]))

            t0 += batch_size
            t1 += batch_size

            yield batch

    res_gen = {}
    res_gen['nb_samples']=nb_samples
    res_gen['nb_batches']=nb_batches
    res_gen['img']=_make_generator(valid_img)
    res_gen['lab']=_make_generator(valid_lab)
    return res_gen


def flow_from_folder_kshot(path_to_folder,
                           batch_size=64,
                           padding=None,
                           sbjt_start_idx=0,
                           meta_batch_size=13,
                           update_batch_size=30
                           ):
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils import get_images
    import cv2

    subjects = os.listdir(path_to_folder)
    subjects.sort()
    subject_folders = [os.path.join(path_to_folder, subject) for subject in subjects]
    folders = subject_folders[sbjt_start_idx:sbjt_start_idx + meta_batch_size]

    inputa_files = []
    inputb_files = []
    labelas = []
    labelbs = []
    # To have totally different inputa and inputb, they should be sampled at the same time and then splitted.
    for sub_folder in folders:  # 쓰일 task수만큼만 경로 만든다. 이 task들이 iteration동안 어차피 반복될거니까
        # random.shuffle(sampled_character_folders)
        off_imgs, on_imgs = get_images(sub_folder, range(2), 0, nb_samples=update_batch_size * 2, validate=False)
        # Split data into a/b
        half_off_img = int(len(off_imgs) / 2)
        half_on_img = int(len(on_imgs) / 2)
        for i in range(half_off_img):
            inputa_files.append(off_imgs[2 * i])
            inputb_files.append(off_imgs[2 * i + 1])
        for i in range(half_on_img):
            inputa_files.append(on_imgs[2 * i])
            inputb_files.append(on_imgs[2 * i + 1])
        label_for_this_subj = [[1, 0]] * half_off_img
        label_for_this_subj.extend([[0, 1]] * half_on_img)
        labelas.extend(label_for_this_subj)
        labelbs.extend(label_for_this_subj)


    print(">>>> inputa_files: ", inputa_files)
    print("--------------------------------------------")
    print(">>>> inputb_files: ", inputb_files)
    print(">>> labelas: ", labelas)
    #################################################################################

    inputa_files.extend(inputb_files)
    labelas.extend(labelbs)
    labelas = np.array(labelas)
    labelas = np.reshape(labelas, (labelas.shape[0], 1, labelas.shape[1]))

    sub = []
    for i in range(len(subjects)): sub.extend([subjects[i]] * update_batch_size * 4)
    print(sub)
    print(">>> img shape: ", np.array(inputa_files).shape)
    print(">>> label shape: ", labelas.shape)
    print(">>> sub shape: ", np.array(sub).shape)
    np.random.seed(1)
    np.random.shuffle(inputa_files)
    np.random.seed(1)
    np.random.shuffle(labelas)
    np.random.seed(1)
    np.random.shuffle(sub)

    f = {'img': np.array(inputa_files), 'lab': labelas, 'sub': np.array(sub)}

    nb_samples = len(inputa_files)
    nb_batches = math.floor(nb_samples / batch_size)
    print('-----------------------------------')
    print('nb_samples: ', nb_samples)
    print('nb_batches: ', nb_batches)
    print('-----------------------------------')

    def _make_generator(data, key):

        t0, t1 = 0, batch_size

        while True:

            t1 = min(nb_samples, t1)  # 배치 사이즈와 샘플 갯수중 작은게 t1
            if t0 >= nb_samples:  # 샘플 갯수보다 to가 크면 변수 초기화
                t0, t1 = 0, batch_size

            if (key == 'img'):
                files = data[t0:t1]
                batch = []
                for file in files:
                    img = cv2.imread(file)
                    batch.append(img)
                batch = np.array(batch)
                print(batch.shape)
            else:
                batch = data[t0:t1]

            if padding != None and batch.shape[0] < batch_size:  # 패딩 설정
                if padding == 'same':
                    batch = data[-batch_size:]
                else:
                    tmp = padding * np.ones([batch_size, *batch.shape[1:]])
                    tmp[:batch.shape[0]] = batch
                    batch = tmp

            t0 += batch_size  # 작업한 배치 사이즈만큼 t0와 t1늘림
            t1 += batch_size

            yield batch  # 패딩이 적용된, 배치사이즈 만큼의 배치데이터 생성

    res_gen = {}
    res_gen['nb_samples'] = nb_samples
    res_gen['nb_batches'] = nb_batches
    for key in f:
        res_gen[key] = _make_generator(f[key], key)

    return res_gen
