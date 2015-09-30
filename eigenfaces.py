import numpy as np

class Eigenfaces:

    def __init__(self, imghandler):
        self.imghandler = imghandler
        calc = self.calculate_data()
        self.eigenfaces = calc[0]
        self.eigenvalues = calc[1]
        self.projected_dataset = calc[2]


    def calculate_data(self):
        try:
            eigenvectors, eigenvalues, coeff_mat = self.imghandler.load_eigenspace_and_coeffmat()
            return (eigenvectors, eigenvalues, coeff_mat)
        except Exception, dirToSave:
            eigenvectors, eigenvalues = self.pca()
            coeff_mat = self.project_training_dataset(eigenvectors)
            np.save(str(dirToSave) + "eigenvectors.npy", eigenvectors)
            np.save(str(dirToSave) + "eigenvalues.npy", eigenvalues)
            np.save(str(dirToSave) + "coefficientmatrix.npy", coeff_mat)
            return (eigenvectors, eigenvalues, coeff_mat)

    def pca(self):
        diff_faces = self.imghandler.diffFaces
        cov_matrix = np.dot(diff_faces,diff_faces.T)
        [eigenvalues, eigenvectors] = np.linalg.eigh(cov_matrix)
        eigenvectors = np.dot(diff_faces.T,eigenvectors)
        for i in xrange(len(self.imghandler.dataset)):
            eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])
        idx = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]
        numimg = len(self.imghandler.dataset)
        eigenvalues = eigenvalues[0:numimg].copy()
        eigenvectors = eigenvectors[:,0:numimg].copy()
        eigenvectors = eigenvectors.T
        return (eigenvectors, eigenvalues)

    def project(self, facevec, eigenvectors):
        diff_face = facevec - self.imghandler.meanface
        coefs = []
        for i in xrange(len(eigenvectors)):
            coefs.append((np.dot(diff_face, eigenvectors[i])))
        return np.asarray(coefs)

    def project_training_dataset(self, eigenvectors):
        coef_matrix = []
        for i in xrange(len(self.imghandler.dataset)):
            coef_matrix.append(self.project(self.imghandler.dataset[i], eigenvectors))
        return np.asarray(coef_matrix)

    def getMinDistanceIndex(self, vector):
        proj = self.project(vector, self.eigenfaces)
        distances = []
        for v in self.projected_dataset:
            distances.append(np.linalg.norm(proj-v))
        return distances.index(min(distances))

    def findMatch(self, img):
        return self.imghandler.getMatchId(self.getMinDistanceIndex(img))
