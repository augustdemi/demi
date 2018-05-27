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
        padding = None 
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
    for key in f: # f는 h5py.File(path_to_file) 데이터 전체이고, 이중 key는 각 그룹을 의미
        res_gen[key] = _make_generator(f[key]) #f[key]는 각 그룹별 데이터 / _make_generator의 결과는 한그룹데이터로 만든 배치 데이터 덩어리들
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
