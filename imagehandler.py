import numpy as np

class Imagehandler:
    def __init__(self, fhandler):
        self.fhandler = fhandler
        self.dataset, self.names  = fhandler.gen_dataset_matrix()
        self.meanface = self.get_mean_face()
        self.diffFaces = self.get_diff_faces()

    def get_mean_face(self):
        return np.mean(self.dataset,0) 
	
	
    def get_diff_faces(self):
        diff_faces = self.dataset - self.meanface
        return diff_faces

    def getMatchId(self, idx):
        return self.names[idx]

    def load_eigenspace_and_coeffmat(self):
        try:
            names = self.fhandler.load_names()
            if (set(self.names) != set(names)):
                raise Exception(self.fhandler.rootdir)
            eigenvectors, eigenvalues, ceoff_mat = self.fhandler.load_eigenspace_and_coeffmat()
            return [eigenvectors, eigenvalues, ceoff_mat]
        except:
            print("Must update training dataset...")
            self.fhandler.save_names(self.names)
            raise Exception(self.fhandler.rootdir)
