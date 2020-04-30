import math
import numpy as np
import pickle
import scipy.io as sio

import open3d

def remove_inner_layer(skull):
    """
    Remove inner layer of skull model to only obtain outer surface

    Note: quality of removal depends on tuning distance and radius variables.
          Unsure if there is way to find optimal values.
    Refer to paper noted in documentation of hidden_point_removal function in
    Open3D.

    Args:
        None

    Returns:
        None
    """
    diameter = np.linalg.norm(np.asarray(skull.get_max_bound()) -
        np.asarray(skull.get_min_bound()))

    theta = [0, 45, 90, 135, 180, 225, 270, 315]
    phi = [0, 45, 90, 135, 180]
    camera = []
    distance = diameter * 3
    for i in range(len(theta)):
        for j in range(len(phi)):
            camera.append([distance*math.sin(math.radians(phi[j]))*math.cos(math.radians(theta[i])),
                           distance*math.sin(math.radians(phi[j]))*math.sin(math.radians(theta[i])),
                           distance*math.cos(math.radians(phi[j]))])

    pts = []
    radius = diameter * 100
    for i in range(len(camera)):
        camera_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
            size=diameter / 5, origin=camera[i])
        pcd, _ = skull.hidden_point_removal(camera[i], radius)
        pts.extend(pcd.vertices)
    skull.points = open3d.utility.Vector3dVector(np.asarray(pts))
    trimesh = open3d.geometry.TriangleMesh()
    trimesh.vertices = skull.points
    trimesh = trimesh.remove_duplicated_vertices()
    skull.points = trimesh.vertices
    return skull

def convert_to_single_layer(skulls, num_voxels):
    train = skulls
    labels = skulls

    for i in range(len(skulls)):
        skull = skulls[i,:,:,:]
        pts = []
        for j in range(num_voxels):
            for k in range(num_voxels):
                for l in range(num_voxels):
                    if skull[j,k,l] != 0:
                        train[i,j,k,l] = 1
                        pts.append([j-int(num_voxels/2),
                                    k-int(num_voxels/2),
                                    l-int(num_voxels/2)])

        skull_pcd = open3d.geometry.PointCloud()
        skull_pcd.points = open3d.utility.Vector3dVector(pts)
        skull_pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # open3d.visualization.draw_geometries([skull_pcd])
        skull_1layer = remove_inner_layer(skull_pcd)
        print(len(skull_1layer.points))

        count = 0
        removed = []
        for pt in pts:
            if not any(np.all([pt[0],pt[1],pt[2]] == l) for l in np.asarray(skull_1layer.points)):
                count = count + 1
                skull[pt[0]+int(num_voxels/2),pt[1]+int(num_voxels/2),pt[2]+int(num_voxels/2)] = 0
                train[i,pt[0]+int(num_voxels/2),pt[1]+int(num_voxels/2),pt[2]+int(num_voxels/2)] = 0
                removed.append([pt[0]+int(num_voxels/2),pt[1]+int(num_voxels/2),pt[2]+int(num_voxels/2)])
        labels[i,:,:,:] = skull
        print(count)

        # skull_pcd.points = open3d.utility.Vector3dVector(removed)
        # open3d.visualization.draw_geometries([skull_pcd])
    return train, labels

labels30 = sio.loadmat("labels30.mat")
skulls30 = labels30['newlabels']

train30, labels30 = convert_to_single_layer(skulls30, 30)
pickle.dump(train30, open( "train30.p", "wb" ) )
pickle.dump(labels30, open( "labels30.p", "wb" ) )
