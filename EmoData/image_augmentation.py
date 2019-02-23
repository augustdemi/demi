import numpy as np
from skimage import transform

def rotate(shape, deg):
    '''
    '''
    # Shift the image so that the centre is around the origin
    center = np.array([int(np.floor(i)/2) for i in shape[:2][::-1]])
    trans = transform.SimilarityTransform(translation=-center)

    # Rotate 
    trans += transform.SimilarityTransform(rotation=np.pi/180*deg)

    # Shift the image back
    trans += transform.SimilarityTransform(translation=center)
    return trans 

def zoom(shape, scale):
    '''
    '''
    # Shift the image so that the centre is around the origin
    center = np.array([int(np.floor(i)/2) for i in shape[:2][::-1]])
    trans = transform.SimilarityTransform(translation=-center)

    # Scale 
    trans += transform.SimilarityTransform(scale=1/(scale+1))

    # Shift the image back
    trans += transform.SimilarityTransform(translation=center)
    return trans 

def width_shift(shape, shift):
    '''
    '''
    return transform.SimilarityTransform(translation=[shift*shape[1],0])

def height_shift(shape, shift):
    '''
    '''
    return transform.SimilarityTransform(translation=[0,shift*shape[0]])
