from os import listdir
from os.path import isfile, join
import pickle

from Mesh import *

# m = MeshFromSTL("./STL/CRANIAL HEADS_Head_1_001.stl")
# m.remove_inner_layer(m.skull)
# m.extract_labeled_data()

path = "/Users/shinaushin/Documents/MATLAB/JHU/Spring 2020/Research/skull-complete/data-prep/matlab-scripts/120/brain_mats"
brain_files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
brain_files.sort()

path = "/Users/shinaushin/Documents/MATLAB/JHU/Spring 2020/Research/skull-complete/data-prep/matlab-scripts/120/scalp_mats"
scalp_files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
scalp_files.sort()

path = "/Users/shinaushin/Documents/MATLAB/JHU/Spring 2020/Research/skull-complete/data-prep/matlab-scripts/120/skull_mats"
skull_files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
skull_files.sort()

train_data = []
# labels = []
j = 18 # 250-277 incl - [1,3] U 278 [4,:) U 565
for i in range(550, 565): # len(brain_files)):
    m = MeshFromMAT(brain_mat_path=brain_files[i], scalp_mat_path=scalp_files[i], skull_mat_path=skull_files[i])
    train, label = m.extract_labeled_data()
    train_data.append(train)
    # labels.append(label)

    print(str(i+1) + "/" + str(len(brain_files)) + ": " + str(brain_files[i][-10:]))

    # if (i+1) % 50 == 0:
    #     print("SAVED")
    #     pickle.dump(train_data, open( "train120_" + str(j) + ".p", "wb" ) )
    #     # pickle.dump(labels, open( "labels120_" + str(j) + ".p", "wb" ) )
    #     j = j + 1
    #     train_data = []
    #     # labels = []

print("SAVED")
pickle.dump(train_data, open( "train120_mid2.p", "wb" ) )
