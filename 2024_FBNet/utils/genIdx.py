import numpy as np
import h5py
import pickle

dataPTH = "../../../Dataset/Deepsig2018/GOLD_XYZ_OSC.0001_1024.hdf5"
Xd = h5py.File(dataPTH,'r')

n_examples = Xd['X'].shape[0]
n_train = int(n_examples * 0.6)
n_val = int(n_examples * 0.2)

train_idx = list(np.random.choice(range(0, n_examples), size=n_train, replace=False))
val_idx   = list(np.random.choice(list(set(range(0,n_examples))-set(train_idx)), size=n_val, replace=False))
test_idx  = list(set(range(0, n_examples)) - set(train_idx)-set(val_idx))

savePTH = "../resource/rml2018_idx.pkl"
self_list = {"train":train_idx,
             "val":val_idx,
             "test":test_idx}
with open(savePTH, 'wb') as f:
    pickle.dump(self_list,f)
    f.close()

