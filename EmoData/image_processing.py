import numpy as np
import os
import h5py
from skimage import transform, draw
from skimage.io import imsave
from skimage.color import gray2rgb

facial_points = {   'inner' : np.arange(17,68),
        'outer' : np.arange(0,27),
        'stable': np.array([36,39,42,45,33]),
        'Stable': np.array([19, 22, 25, 28, 10, 11, 12, 13, 14, 15, 16, 17, 18])+17,
        'all'   : np.arange(68)}
print(os.path.dirname(__file__)+'/data/mean_shape.h5')
with h5py.File(os.path.dirname(__file__)+'/data/mean_shape.h5') as tmp:
    ss = list(tmp.keys())
    mean_shape = tmp['data'][::]
    connections = tmp['connections'][::]

def add_landmarks_to_img(img, pts):
    radius = max(pts.max(0)-pts.min(0))/60
    max_val = np.max(img)
    if len(img.shape)==2:img = gray2rgb(img)
    for x,y in np.int16(pts):
        rr, cc = draw.circle(y, x, radius)
        try:
            img[rr, cc, 0] = max_val
            img[rr, cc, 1] = max_val
            img[rr, cc, 2] = 0
        except IndexError:
            pass

    for p0, p1 in connections:
        x = np.int16(pts[p0])
        y = np.int16(pts[p1])

        rr, cc, val = draw.line_aa(x[1],x[0],y[1],y[0])
        try:
            img[rr, cc, 1] = max_val
        except IndexError:
            pass

    return img
        
def bbox_extractor(
        detector, 
        shape_predictor, 
        img, 
        output_size = [128, 96],
        face_size = 96,
        allignment_type = 'similarity',
        fill_mode = 'edge',
        fix_pts ='all',
        mean_shape = mean_shape,
        ):

    ################################################################
    # scale mean shape to face size size
    ################################################################

    # move to origin
    mean_shape[0]-=mean_shape.mean(1)[0]
    mean_shape[1]-=mean_shape.mean(1)[1]

    # scale mean_shape to face size
    scale = max(mean_shape.max(1)-mean_shape.min(1))
    mean_shape*=face_size/scale

    # move back to center 
    mean_shape[0]+=output_size[0]/2
    mean_shape[1]+=output_size[1]/2
    ################################################################




    ################################################################
    # detect bbox and facial points
    ################################################################
    tmp = np.float32(np.copy(img))
    tmp -= tmp.min()
    tmp /= tmp.max()
    tmp = np.uint8(tmp*255)
    rect = detector(tmp, 0)

    # if tracking fails: return zeros
    if len(rect)==0:
        img_null = np.zeros([output_size[1],output_size[0],img.shape[-1]])
        pts_null = np.zeros_like(mean_shape.T)
        return img_null, pts_null, pts_null

    ################################################################
    # get landmarks form bbox
    ################################################################
    bbox = rect[0]
    shape = shape_predictor(tmp, bbox)
    pts = np.vstack([ [p.x, p.y] for p in shape.parts() ])
    ################################################################

    
    ################################################################
    # make the alignment with respect to points from mean_shape
    ################################################################

    pts_for_alignment = facial_points[fix_pts]

    # if self.allignment_type==None:
    # implement here only translation of input image 

    src = mean_shape.T[pts_for_alignment]
    dst = pts[pts_for_alignment]
    trans = transform.estimate_transform(allignment_type, src, dst)

    # scale input image 
    # (skimage requires values between -1 and 1 for the transformation)
    min_val, max_val = img.min(), img.max()
    img = (img-min_val)/float(max_val)

    img = transform.warp(
            img, 
            inverse_map = trans, 
            mode=fill_mode, 
            output_shape=output_size[::-1] 
            )

    # scale it back
    box = (img*max_val)+min_val

    # apply transformation on facial points
    pts_box = trans.inverse(pts)
    ################################################################

    return box, pts_box, pts

def save_image(img, pwd):
    '''
    '''
    if len(img.shape)==2:img=gray2rgb(img)
    img = np.float32(img)
    img-=img.min()
    img/=img.max()
    if img.shape[-1]==1:img=img[:,:,0]
    imsave(pwd,img)
