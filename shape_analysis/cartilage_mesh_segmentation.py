#!/usr/bin/env python3.6
"""
TODO: The script try to segment the cartilage surface mesh up/down side with clustering algorithm

Created by zhenlinx on 7/18/18
"""
import matplotlib

matplotlib.use('Qt5Agg')

# from pathos.multiprocessing import ProcessingPool as Pool

from shape_analysis.cartilage_shape_processing import *
# Though the following import is not directly being used, it is required
# for 3D projection to work

import visvis as vv
from sklearn.cluster import KMeans


def split_surface_with_clustering(mesh):
    pass

def vertex_attribute_to_face(mesh, vertex_attribute):
    assert mesh.num_vertices == vertex_attribute.shape[0],\
        "The first demension of vertex_attributes have to match the number of vertices in mesh"
    face_attributes = []
    for k in range(mesh.num_faces):
        face_attributes.append(np.mean(vertex_attribute[mesh.faces[k, :], ...], axis=0))
    return np.stack(face_attributes, axis=0)

def compute_curvature_per_point(point, neighbor_points):
    """
    compute curvature by mean((n_j - n_0)*(p_j - p_0)/|p_j - p_0|)
    :param points (1x6 array): [position, normal] of point where the curvature is computed
    :param neighbor_points (Nx6 array): [positions, normals] neighbor points
    :return: mean curvature
    """
    temp_curvatures = []
    for neighbor_point in neighbor_points:
        temp_curvatures.append(np.dot(neighbor_point[3:] - point[3:], np.linalg.norm(neighbor_point[:3] - point[:3])))
    mean_curvature = np.mean(temp_curvatures)
    return mean_curvature


def get_curvature_per_process(id, element, kernal_size=1, edge_only=False):
    """
    compute mean curvature of face_id over region of kernal_size
    :param id: id of element where curvature is computed
    :param element: 'v'/'vertex' for vertex, or 'f'/'face' for face
    :param kernal_size: size of region where the curvature is computed
    :return: curvature value
    """
    neighbor_indices = get_neighbors(MESH, id, element=element, search_range=kernal_size, edge_only=edge_only)
    if len(neighbor_indices) == 0:
        print("Warning: no neighboring points of id {} at distance {}".format(id, kernal_size))
        return 0
    else:
        center_point = np.concatenate((POSITIONS[id, :], NORMALS[id, :]))
        neighbor_points = np.concatenate((POSITIONS[neighbor_indices, :], NORMALS[neighbor_indices, :]), axis=1)

        return compute_curvature_per_point(center_point, neighbor_points)


def get_curvature_parallel(mesh, element, kernal_size=1, n_workers=1, edge_only=False):
    """
    compute mean curvature of faces over region of kernal_size with multiprocess
    :param mesh: a PyMesh mesh object
    :param element: 'v'/'vertex' for vertex, or 'f'/'face' for face
    :param kernal_size: size of region where the curvature is computed
    :param n_workers: number of processes
    :return: a vector of the size of number of faces
    """

    def pool_init(mesh, positions, normals):
        global MESH, POSITIONS, NORMALS
        MESH = mesh
        POSITIONS = positions
        NORMALS = normals

    if element == 'v':
        num_element = mesh.num_vertices
        mesh.add_attribute('vertex_normal')
        positions = mesh.vertices
        normals = mesh.get_attribute('vertex_normal').reshape(-1, 3)

    elif element == 'f':
        num_element = mesh.num_faces
        mesh.add_attribute("face_centroid")
        mesh.add_attribute("face_normal")
        positions = mesh.get_attribute('face_centroid').reshape(-1, 3)
        normals = mesh.get_attribute('face_normal').reshape(-1, 3)
    else:
        ValueError("Wrong Element Type: can only be  'v'/'vertex' for vertex, or 'f'/'face' for face")

    with Pool(processes=n_workers, initializer=pool_init, initargs=(mesh, positions, normals)) as pool:
        curvatures = pool.map(partial(get_curvature_per_process, element=element, kernal_size=kernal_size,
                                      edge_only=edge_only), range(num_element))
    return np.array(curvatures)


def get_vertex_curvature(mesh, kernal_size=1):
    """
    compute mean curvature of faces over region of kernal_size
    :param mesh: a PyMesh mesh object
    :param kernal_size: size of region where the curvature is computed
    :return: a vector of the size of number of faces
    """
    vertex_curvatures = np.zeros(mesh.num_vertices)

    mesh.add_attribute('vertex_normal')
    vertex_positions = mesh.vertices
    vertex_normals = mesh.get_attribute('vertex_normal').reshape(-1, 3)

    for vertex_id in range(mesh.num_vertices):
        neighbor_vertex_indices = get_neighbors(mesh, vertex_id, 'vertex', search_range=kernal_size)
        center_vertex = np.concatenate((vertex_positions[vertex_id, :], vertex_normals[vertex_id, :]))
        neighbor_vertex = np.concatenate((vertex_positions[neighbor_vertex_indices, :],
                                          vertex_normals[neighbor_vertex_indices, :]), axis=1)
        vertex_curvatures[vertex_id] = compute_curvature_per_point(center_vertex, neighbor_vertex)
    return vertex_curvatures


def get_face_curvature(mesh, kernal_size=1):
    """
    compute mean curvature of faces over region of kernal_size
    :param mesh: a PyMesh mesh object
    :param kernal_size: size of region where the curvature is computed
    :return: an array of the size of number of faces
    """
    face_curvatures = np.zeros(mesh.num_faces)

    mesh.add_attribute("face_centroid")
    mesh.add_attribute("face_normal")
    face_centroids = mesh.get_attribute('face_centroid').reshape(-1, 3)
    face_normals = mesh.get_attribute('face_normal').reshape(-1, 3)

    for face_id in range(mesh.num_faces):
        neighbor_face_indices = get_neighbors(mesh, face_id, element='f', search_range=kernal_size)
        center_face = np.concatenate((face_centroids[face_id, :], face_normals[face_id, :]))
        neighbor_faces = np.concatenate(
            (face_centroids[neighbor_face_indices, :], face_normals[neighbor_face_indices, :]), axis=1)
        face_curvatures[face_id] = compute_curvature_per_point(center_face, neighbor_faces)

    return face_curvatures


def get_curvature(mesh, element, kernal_size=1):
    """
    compute curvature of face/vertex on a mesh over region of kernal_size
    :param mesh: a PyMesh mesh object
    :param element: 'v'/'vertex' for vertex, or 'f'/'face' for face
    :param kernal_size: size of region where the curvature is computed
    :return: an array of the size of number of faces
    """
    if element == 'v':
        num_element = mesh.num_vertices
        curvatures = np.zeros(num_element)
        mesh.add_attribute('vertex_normal')
        positions = mesh.vertices
        normals = mesh.get_attribute('vertex_normal').reshape(-1, 3)

    elif element == 'f':
        num_element = mesh.num_faces
        curvatures = np.zeros(num_element)
        mesh.add_attribute("face_centroid")
        mesh.add_attribute("face_normal")
        positions = mesh.get_attribute('face_centroid').reshape(-1, 3)
        normals = mesh.get_attribute('face_normal').reshape(-1, 3)
    else:
        ValueError("Wrong Type Input: can only be  'v'/'vertex' for vertex, or 'f'/'face' for face")

    for id in range(num_element):
        neighbor_indices = get_neighbors(mesh, id, element=element, search_range=kernal_size)
        center_point = np.concatenate((positions[id, :], normals[id, :]))
        neighbor_points = np.concatenate((positions[neighbor_indices, :], normals[neighbor_indices, :]), axis=1)
        curvatures[id] = compute_curvature_per_point(center_point, neighbor_points)

    return curvatures


def normalize_array(input):
    return input - input.mean() / input.std()


def main():
    mesh = pymesh.load_mesh('FC_mesh_main.ply')
    # mesh = pymesh.load_mesh('FC_inverse_meshlab.ply')
    # mesh = pymesh.load_mesh('TC_mesh.ply')

    mesh.enable_connectivity()

    # mesh = get_largest_mesh(pymesh.separate_mesh(mesh), 1)
    mesh.add_attribute("face_centroid")
    mesh.add_attribute("face_normal")

    mesh_centroids = mesh.get_attribute('face_centroid').reshape(-1, 3)
    mesh_centroids_normalized = (mesh_centroids - np.mean(mesh_centroids, axis=0)) / \
                                (np.max(mesh_centroids, axis=0) - np.min(mesh_centroids, axis=0))
    mesh_normals = mesh.get_attribute('face_normal').reshape(-1, 3)

    # curvature
    mesh.add_attribute("vertex_mean_curvature")
    mesh.add_attribute("vertex_gaussian_curvature")
    vertex_mean_curvature = np.copy(mesh.get_attribute('vertex_mean_curvature'))
    vertex_gaussian_curvature = np.copy(mesh.get_attribute('vertex_gaussian_curvature'))

    # my_vertex_curvature = get_curvature_parallel(mesh, element='v', kernal_size=9, n_workers=16, edge_only=False)
    # my_face_curvature = get_curvature_parallel(mesh, element='v', kernal_size=9, n_workers=10)

    # my_vertex_curvature = normalize_array(my_vertex_curvature)
    # vertex_mean_curvature = normalize_array(vertex_mean_curvature)
    # vertex_gaussian_curvature = normalize_array(vertex_gaussian_curvature)

    # clip_threshold = [-0.1, 0.1]
    # vertex_mean_curvature_clipped = np.clip(vertex_mean_curvature, 4*clip_threshold[0], 4*clip_threshold[1])
    # vertex_gaussian_curvature_clipped = np.clip(vertex_gaussian_curvature, clip_threshold[0], clip_threshold[1])
    # my_vertex_curvature_clipped = np.clip(my_vertex_curvature, clip_threshold[0], clip_threshold[1])
    #
    # my_vertex_curvature = my_vertex_curvature/my_vertex_curvature.std()

    # app = vv.use()
    # a1 = vv.subplot(311)
    # mean_curv_vis = vv.mesh(mesh.vertices, mesh.faces, values=np.sign(vertex_mean_curvature))
    # mean_curv_vis.colormap = vv.CM_JET
    # vv.colorbar()
    # a2 = vv.subplot(312)
    # gauss_curv_vis = vv.mesh(mesh.vertices, mesh.faces, values=np.sign(vertex_gaussian_curvature))
    # gauss_curv_vis.colormap = vv.CM_JET
    # vv.colorbar()
    # a3 = vv.subplot(313)
    # my_curv_vis = vv.mesh(mesh.vertices, mesh.faces, values=np.sign(my_vertex_curvature))
    # my_curv_vis.colormap = vv.CM_JET
    # vv.colorbar()
    # app.Run()
    #
    # app = vv.use()
    # a1 = vv.subplot(311)
    # mean_curv_vis = vv.mesh(mesh.vertices, mesh.faces, values=vertex_mean_curvature)
    # mean_curv_vis.colormap = vv.CM_JET
    # vv.colorbar()
    # a2 = vv.subplot(312)
    # gauss_curv_vis = vv.mesh(mesh.vertices, mesh.faces, values=vertex_gaussian_curvature_clipped)
    # gauss_curv_vis.colormap = vv.CM_JET
    # vv.colorbar()
    # a3 = vv.subplot(313)
    # my_curv_vis = vv.mesh(mesh.vertices, mesh.faces, values=my_vertex_curvature_clipped)
    # my_curv_vis.colormap = vv.CM_JET
    # vv.colorbar()
    # app.Run()

    # my_face_curvature = get_curvature_parallel(mesh, element='f', kernal_size=8, n_workers=12)
    vertex_mean_curvature[np.isnan(vertex_mean_curvature)] = 0
    vertex_gaussian_curvature[np.isnan(vertex_gaussian_curvature)] = 0
    # my_face_curvature[np.isnan(my_face_curvature)] = 0
    face_mean_curvature = vertex_attribute_to_face(mesh, vertex_mean_curvature).reshape(-1,1)

    clip_threshold = [-0.1, 0.1]
    face_mean_curvature = np.clip(face_mean_curvature, clip_threshold[0], clip_threshold[1])
    # vertex_mean_curvature = np.clip(vertex_mean_curvature, clip_threshold[0], clip_threshold[1])
    # vertex_gaussian_curvature = np.clip(vertex_gaussian_curvature, clip_threshold[0], clip_threshold[1])

    # vertex_mean_curvature_feature = normalize_array(vertex_mean_curvature)
    # vertex_gaussian_curvature_feature = (vertex_gaussian_curvature - vertex_gaussian_curvature.mean())/np.std(vertex_gaussian_curvature)
    # my_face_curvature_feature = normalize_array(my_face_curvature).reshape(-1,1)

    features = np.concatenate((mesh_centroids_normalized*3, mesh_normals*0, face_mean_curvature*10), axis=1)
    est = KMeans(n_clusters=2, max_iter=1000)
    # est = SpectralClustering(n_clusters=2)
    labels = est.fit(features).labels_

    inner_outer_label_list = labels*2-1

    mesh.enable_connectivity()
    inner_mesh, outer_mesh, _, _ = smooth_mesh_segmentation(mesh, inner_outer_label_list, smooth_rings=10, max_rings=10, n_workers=20)


    # inner_mesh = pymesh.submesh(mesh, np.where(labels == 0)[0], num_rings=0)
    # outer_mesh = pymesh.submesh(mesh, np.where(labels == 1)[0], num_rings=0)

    app = vv.use()
    a1 = vv.subplot(311)
    FC_vis_up = vv.mesh((inner_mesh.vertices), inner_mesh.faces)
    FC_vis_up.faceColor = 'r'
    FC_vis_down = vv.mesh((outer_mesh.vertices), outer_mesh.faces)
    FC_vis_down.faceColor = 'b'

    a2 = vv.subplot(312)
    FC_vis2 = vv.mesh((inner_mesh.vertices), inner_mesh.faces)
    FC_vis2.faceColor = 'r'

    a3 = vv.subplot(313)
    FC_vis3 = vv.mesh((outer_mesh.vertices), outer_mesh.faces)
    FC_vis3.faceColor = 'b'
    app.Run()


if __name__ == '__main__':
    main()
