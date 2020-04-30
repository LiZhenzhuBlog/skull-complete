import numpy as np
import pickle

mat_files = ["train120_1.p", "train120_2.p", "train120_3.p", "train120_mid1.p",
            "train120_4.p", "train120_5.p", "train120_6.p", "train120_7.p", 
            "train120_8.p", "train120_9.p", "train120_mid2.p", "train120_10.p", 
            "train120_11.p", "train120_12.p", "train120_13.p", "train120_14.p", 
            "train120_15.p", "train120_16.p", "train120_17.p", "train120_end.p", ]

# Bad Training Data
mat120_1 = [1, 4, 5, 17, 36, 46]
mat120_2 = [2, 25, 28, 110, 117, 120] 
mat120_3 = [30, 36]
mat120_mid1 = [18]
mat120_4 = []
mat120_5 = [3, 26]
mat120_6 = [20, 28, 33]
mat120_7 = []
mat120_8 = [11, 37, ]
mat120_9 = [9, 21, 37, 45]
mat120_mid2 = []
mat120_10 = [30]
mat120_11 = [19]
mat120_12 = [1, 11, 14, 20, 39]
mat120_13 = [19, 20, 48]
mat120_14 = [11]
mat120_15 = [3, 11]
mat120_16 = [39]
mat120_17 = [28]
mat120_end = []

mat_indices_fix = [mat120_1, mat120_2, mat120_3, mat120_mid1, mat120_4, mat120_5, 
                mat120_6, mat120_7, mat120_8, mat120_9, mat120_mid2, mat120_10, 
                mat120_11, mat120_12, mat120_13, mat120_14, mat120_15, mat120_16, 
                mat120_17, mat120_end]

# Remove Training Data
mat120_1 = [20]
mat120_2 = [3, 30, 74, 85, 95, 101, 131]
mat120_3 = [13, 24, 45, 48]
mat120_mid1 = [10] # partial scalp - 1
mat120_4 = []
mat120_5 = [32, 37, ] # partial scalp - 39
mat120_6 = [23]
mat120_7 = [33, 43, 44, 45]
mat120_8 = [12, 20, 23, 33, 39]
mat120_9 = [40, ]
mat120_mid2 = []
mat120_10 = [5, 9]
mat120_11 = [1, 5]
mat120_12 = [2, 19, 48]
mat120_13 = []
mat120_14 = [0, 26, 37]
mat120_15 = [49]
mat120_16 = [20, 32]
mat120_17 = [16, 17, 22, 25, 39]
mat120_end = [4]

mat_indices_bad = [mat120_1, mat120_2, mat120_3, mat120_mid1, mat120_4, mat120_5, 
                mat120_6, mat120_7, mat120_8, mat120_9, mat120_mid2, mat120_10, 
                mat120_11, mat120_12, mat120_13, mat120_14, mat120_15, mat120_16, 
                mat120_17, mat120_end]

total_indices = []
for i in range(len(mat_indices_bad)):
    idx = []
    idx.extend(mat_indices_fix[i])
    idx.extend(mat_indices_bad[i])
    idx.sort(reverse=True)
    total_indices.append(idx)

# print(total_indices)

mat = []
labels = pickle.load( open("labels120.p", "rb" ) )
final_labels = []

train_fix = []
labels_fix = []

for i in range(len(mat_files)):
    train = pickle.load( open(mat_files[i], "rb" ) )
    length = np.asarray(train).shape[0]
    label_subset = labels[:length]
    labels = labels[length:]

    idx_fix = mat_indices_fix[i]
    for j in range(len(idx_fix)):
        train_fix.append(train[idx_fix[j]])
        labels_fix.append(label_subset[idx_fix[j]])

    idx = total_indices[i]
    # print(np.asarray(train).shape)
    for j in range(len(idx)):
        del train[idx[j]]
        del label_subset[idx[j]]
    # print(np.asarray(train).shape)

    for j in range(np.asarray(train).shape[0]):
        mat.append(train[j])
        final_labels.append(label_subset[j])

print(np.asarray(mat).shape)
print(np.asarray(final_labels).shape)
print(np.asarray(train_fix).shape)
print(np.asarray(labels_fix).shape)

pickle.dump(mat, open( "train120_final.p", "wb" ) )
pickle.dump(train_fix, open( "train120_fix.p", "wb" ) )
pickle.dump(final_labels, open( "labels120_final.p", "wb" ) )
pickle.dump(labels_fix, open( "labels120_fix.p", "wb" ) )
