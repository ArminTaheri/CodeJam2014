import os
from PIL import Image
import numpy as np


class Filehandler:
    
    def __init__(self, rootdir):
        self.rootdir = rootdir
        self.dataset_matrix, self.picture_indices = get_dataset_matrix(rootdir)

    def get_dataset_matrix(rootdir):
        files, dirs = os.walk(rootdir)
        pic_idx = []
        im_list = []
        i = 0
        for f in files[2]:
            im = Image.open(rootdir + f)
            im = im.convert("L")
            imvec = np.asarray(im, dtype=np.uint8)
            imvec = imvec.reshape(imvec.shape[0]*imvec.shape[1])
            im_list.append(imvec)
            pic_idx.append(i++)
        return im_list, pic_idx
    def 
