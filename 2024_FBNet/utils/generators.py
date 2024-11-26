from tensorflow import keras
import numpy as np
import h5py
from utils.PreProcess import FilterBank32

class genRML2018(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, pth, list_IDs, batch_size=32, dim=(1024,), n_channels=2,
                 n_classes=24, shuffle=True ,type = 'lstm'):
        'Initialization'
        self.pth = pth
        self.dim = dim        # dimension
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels # each dimension's channel
        self.n_classes = n_classes   # Y's classes
        self.shuffle = shuffle
        self.type = type
        # generation
        self.on_epoch_end()

    def __len__(self):
        # 重写len方法,在本模块中调用len就会返回这里定义的内容
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)
        return X, Y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, self.n_classes))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            Xd = h5py.File(self.pth)
            # Filter Bank 32
            X_tmp = np.zeros(*self.dim, self.n_channels)
            X_tmp = Xd['X'][ID,0:self.dim[0]]
            X_tmp = FilterBank32(X_tmp,flat=True)
            X[i,] = X_tmp       # aim at IQ signals
            # Store class
            Y[i] = Xd['Y'][ID]

        if self.type == 'lstm' :          
            return X,Y                      # batch*1024*2
        elif self.type == 'cnn':
            return X.transpose(0,2,1), Y    # batch*2*1024

class genRML2018_FB32(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, pth, list_IDs, batch_size=32, dim=(1024,), n_channels=2,
                 n_classes=24, shuffle=True ,type = 'lstm'):
        'Initialization'
        self.pth = pth
        self.dim = dim        # dimension
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels # each dimension's channel
        self.n_classes = n_classes   # Y's classes
        self.shuffle = shuffle
        self.type = type
        # generation
        self.on_epoch_end()

    def __len__(self):
        # 重写len方法,在本模块中调用len就会返回这里定义的内容
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)
        return X, Y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, self.n_classes))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            Xd = h5py.File(self.pth)
            X[i,] = Xd['X'][ID,0:self.dim[0]]       # aim at IQ signals
            # Store class
            Y[i] = Xd['Y'][ID]

        if self.type == 'lstm' :          
            return X,Y                      # batch*1024*2
        elif self.type == 'cnn':
            return X.transpose(0,2,1), Y    # batch*2*1024
