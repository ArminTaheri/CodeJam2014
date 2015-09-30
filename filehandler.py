import os
import re
import pickle
from PIL import Image
import numpy as np


class Filehandler:
    
    def __init__(self, rootdir):
        self.rootdir = rootdir

    def gen_dataset_matrix(self):
        for root, dirs, files in os.walk(self.rootdir):
		pic_names = []
		im_list = []
		for f in files:
		    reg = re.search('(.gif)|(.bmp)', f)
		    if reg:
		        im = Image.open(self.rootdir + f)
		        im = im.convert("L")
		        imvec = np.asarray(im, dtype=np.uint8)
		        imvec = imvec.reshape(imvec.shape[0]*imvec.shape[1])
		        im_list.append(imvec)
		        pic_names.append(f)
		return (im_list, pic_names)

    def open_image(self,imgfile):
        im  = Image.open(imgfile)
        imv = np.asarray(im, dtype=np.uint8)
        imv = imv.reshape(imv.shape[0] * imv.shape[1])
        return  imv

    def load_eigenspace_and_coeffmat(self):
        eigenvectors  = np.load(self.rootdir + "eigenvectors.npy")
        eigenvalues   = np.load(self.rootdir + "eigenvalues.npy")
        coeff_mat     = np.load(self.rootdir + "coefficientmatrix.npy")
        return (eigenvectors, eigenvalues, coeff_mat)

    def load_names(self):
        f = open(self.rootdir + "pictureNames.nm", "r")
        if f:
            names = pickle.load(f)
            f.close()
            return names
        else:
            return []

    def save_names(self, namelist):
        f = open(self.rootdir + "pictureNames.nm", "w")
        pickle.dump(namelist, f)
        f.close()

