import numpy as np
import pickle
from random import randrange
import scipy.io

train = pickle.load( open( "train120_final.p", "rb" ) )
labels = pickle.load( open( "labels120_final.p", "rb" ) )
train = np.asarray(train)
labels = np.asarray(labels)
total = train.shape[0]
# print(total)

rand_nums = np.random.choice(total, total, replace=False)

test_idx = rand_nums[:173]
valid_idx = rand_nums[173:312]
train_idx = rand_nums[312:]

# test set - 173
test_x = train[test_idx,:,:,:]
# print(test_x.shape)
test_y = labels[test_idx,:,:,:]

# train set
train_x = train[train_idx,:,:,:]
# print(train_x.shape)
train_y = labels[train_idx,:,:,:]

# validation set - 139
valid_x = train[valid_idx,:,:,:]
# print(valid_x.shape)
valid_y = labels[valid_idx,:,:,:]

np.save("train_x.npy", train_x, allow_pickle=False)
np.save("train_y.npy", train_y, allow_pickle=False)
np.save("test_x.npy", test_x, allow_pickle=False)
np.save("test_y.npy", test_y, allow_pickle=False)
np.save("valid_x.npy", valid_x, allow_pickle=False)
np.save("valid_y.npy", valid_y, allow_pickle=False)
