from random import randrange
import numpy as np
import pickle

import open3d

def mat2pcd(mat):
    coords = []
    for i in range(120):
        for j in range(120):
            for k in range(120):
                if mat[i,j,k] != 0:
                    coords.append([i,j,k])
    coords = np.asarray(coords)
    # print(min(coords[:,0]), max(coords[:,0]), min(coords[:,1]), max(coords[:,1]), min(coords[:,2]), max(coords[:,2]))
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(coords)
    open3d.visualization.draw_geometries([pcd])

def mat2pcd4contour(mats):
    # total = mats.shape[0]
    # for l in range(total):
    #     mat = mats[l,:,:,:]
    #     mat.shape = (120,120,120)
    #     coords = []
    #     for i in range(120):
    #         for j in range(120):
    #             for k in range(120):
    #                 if mat[i,j,k] == 2:
    #                     coords.append([i,j,k])
    #     coords = np.asarray(coords)
    #     if len(coords) == 0:
    #         print(str(l) + ": True")
    #     else:
    #         print(str(l) + ": False")

    contour_coords = []
    contour_colors = []
    head_coords = []
    head_colors = []
    for i in range(120):
        for j in range(120):
            for k in range(120):
                if mats[i,j,k] == 2:
                    contour_coords.append([i,j,k])
                    contour_colors.append([1, 0, 0])
                if mats[i,j,k] == 1:
                    head_coords.append([i,j,k])
                    head_colors.append([0, 0, 1])
    contour_coords = np.asarray(contour_coords)
    head_coords = np.asarray(head_coords)

    contour_pcd = open3d.geometry.PointCloud()
    contour_pcd.points = open3d.utility.Vector3dVector(contour_coords)

    head_pcd = open3d.geometry.PointCloud()
    head_pcd.points = open3d.utility.Vector3dVector(head_coords)

    open3d.visualization.draw_geometries([contour_pcd, head_pcd])

    contour_pcd.colors = open3d.utility.Vector3dVector(contour_colors)
    head_pcd.colors = open3d.utility.Vector3dVector(head_colors)

    open3d.visualization.draw_geometries([contour_pcd, head_pcd])

def plot_train_random(mats):
    total = mats.shape[0]
    # i = randrange(total)
    # print(i)
    for i in range(total):
        print(i)
        mat = mats[i,:,:,:]
        mat.shape = (120,120,120)
        mat2pcd(mat)

def plot_labels_random(mats):
    total = mats.shape[0]
    i = randrange(total)
    mat = mats[i,:,:,:]
    mat.shape = (120,120,120)
    mat2pcd4contour(mat)

# train = np.load( "train120_end.p", "rb" )
labels = np.load( "../trainer/test_y.npy")
print(labels.shape)

# plot_train_random(train)
plot_labels_random(labels)
# mat2pcd4contour(labels)
