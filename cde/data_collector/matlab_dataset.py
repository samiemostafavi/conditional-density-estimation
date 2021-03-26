import scipy.io as sio
import numpy as np
import h5py


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
    def get_data_servicerate_cond(self, s_cond):
        """
        This function returns a vector of n_samples from the dataset randomly
        """

        idx = []
        new_ds = self.dataset[:,[2,4,6]]
        for i in range(len(new_ds)):
            if all(new_ds[i] == s_cond):
                idx.append(i)
        cond_res = self.dataset[idx,:]

        idz = np.random.randint(len(cond_res), size=len(cond_res))
        return cond_res[idz,:]

    def get_data_firstrows_cond(self, n_samples):
        """
        This function returns a vector of n_samples from the dataset randomly
        """

        new_ds = self.dataset[0:n_samples,:]
        idx = np.random.randint(len(new_ds), size=len(new_ds))
        return new_ds[idx,:]

""" load dataset """
class MatlabDatasetH5():
    def __init__(self, file_address):

        f = h5py.File(file_address, 'r')
        #print(f.keys())
        self.dataset = np.transpose(np.array(f['datasetclean']))
        self.n_records = len(self.dataset)
        self.n_features = len(self.dataset[0])
        print(' Dataset H5 loaded from .mat file. Rows: %d ' % len(self.dataset), ' Columns: %d ' % len(self.dataset[0]) )
    def get_data(self, n_samples):
        """
        This function returns a vector of n_samples from the dataset randomly
        """
        idx = np.random.randint(self.n_records, size=n_samples)
        return self.dataset[idx,:]
    def get_data_servicerate_cond(self, s_cond):
        """
        This function returns a vector of n_samples from the dataset randomly
        """

        idx = []
        new_ds = self.dataset[:,[2,4,6]]
        for i in range(len(new_ds)):
            if all(new_ds[i] == s_cond):
                idx.append(i)
        cond_res = self.dataset[idx,:]

        idz = np.random.randint(len(cond_res), size=len(cond_res))
        return cond_res[idz,:]

    def get_data_firstrows_cond(self, n_samples):
        """
        This function returns a vector of n_samples from the dataset randomly
        """

        new_ds = self.dataset[0:n_samples,:]
        idx = np.random.randint(len(new_ds), size=len(new_ds))
        return new_ds[idx,:]