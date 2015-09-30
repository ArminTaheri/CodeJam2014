import os
import numpy as np
import scipy as sp
from PIL import Image
import matplotlib.pyplot as plt

def get_dataset_matrix(rootdir):
    files, dirs = os.walk(rootdir)
    im_list = []
    for f in files[2]:
        im = Image.open(rootdir + f)
        im = im.convert("L")
        imvec = np.asarray(im, dtype=np.uint8)
        imvec = imvec.reshape(imvec.shape[0]*imvec.shape[1])
        im_list.append(imvec)
    return im_list        


def get_mean_face(im_mat):
 return np.mean(im_matrix,0) 


def get_diff_faces(mean_face, face_matrix):
    diff_faces =face_matrix - mean_face
    return diff_faces
    
def pca(diff_faces):
	cov_matrix = np.dot(diff_faces,diff_faces.T)
	[eigenvalues, eigenvectors] = np.linalg.eigh(cov_matrix)
	eigenvectors = np.dot(diff_faces.T,eigenvectors)
	for i in xrange(numpictures):
		eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])
	idx = np.argsort(-eigenvalues)
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:,idx]
	eigenvalues = eigenvalues[0:numpictures].copy()
	eigenvectors = eigenvectors[:,0:numpictures].copy()
	return [eigenvalues,eigenvectors.T]

def projectImg(mean_face,face_img,eigenfaces):
        facevec = np.asarray(face_img, dtype=np.uint8)
        facevec = facevec.reshape(facevec.shape[0]*facevec.shape[1])
	diff_face = facevec - mean_face
	coefs = [];
	for i in range(len(eigenfaces)):
		coefs.append((np.dot(diff_face,eigenfaces[i])))
	return np.asarray(coefs)

def projectVector(mean_face,facevec, eigenfaces):
	diff_face = facevec - mean_face
	coefs = [];
	for i in range(len(eigenfaces)):
		coefs.append((np.dot(diff_face,eigenfaces[i])))
	return np.asarray(coefs)


def project_training_dataset(im_matrix,mean_face, eigenfaces):
	coef_matrix =[]
	for i in range(len(im_matrix)):
		coef_matrix.append(projectVector(mean_face,im_matrix[i],eigenfaces))
	return np.asarray(coef_matrix)


def getMinDistanceIndex(coef_matrix, projection):
	distances = []
	for v in coef_matrix:
		distances.append(np.linalg.norm(projection - v))	 
	return distances.index(min(distances))


'''
def calculate_distance(face_img, mean_face,coef_matrix,eigenfaces):
	facevec = np.asarray(face_img, dtype=np.uint8)
        facevec = facevec.reshape(facevec.shape[0]*facevec.shape[1])
	face_point = project(mean_face,facevec,eigenfaces)
	min = 2**32-1
	minidx = 0
	for i in range(len(coef_matrix[0])):
		if np.sqrt(np.sum(np.power((coef_matrix[i]-face_point),2))) < min:
			minidx = i	
	return minidx;
'''
im_matrix = get_dataset_matrix("tr_dataset/")
global numpictures  
numpictures = len(im_matrix)
mean_face = get_mean_face(im_matrix)
diff_faces = get_diff_faces(mean_face,im_matrix)
[eigenvalues,eigenvectors] = pca(diff_faces)
testImg = Image.open("5_7_.gif")
impro = projectImg(mean_face,testImg,eigenvectors)
coef_matrix = project_training_dataset(im_matrix,mean_face,eigenvectors)
minidx =  getMinDistanceIndex(coef_matrix,impro)
result = im_matrix[minidx]
plt.imshow(result.reshape((243,320)))
plt.gray()
plt.show()
#subjectidx = calculate_distance(testImg, mean_face,coef_matrix, eigenvectors) 
#print subjectidx
