import scipy.io as sio
import numpy as np


""" load dataset """
class MatlabDataset():
    def __init__(self, file_address):
         contents = sio.loadmat(file_address)
         self.dataset = contents['datasetclean']
         self.n_records = len(self.dataset)
         self.n_features = len(self.dataset[0])
         print(' Dataset loaded from .mat file. Rows: %d ' % len(self.dataset), ' Columns: %d ' % len(self.dataset[0]) )
    def get_data(self, n_samples):
        """
        This function returns a vector of n_samples from the dataset randomly
        """
        idx = np.random.randint(self.n_records, size=n_samples)
        return self.dataset[idx,:]