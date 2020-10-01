#!/usr/bin/env python3.6
"""
A class for computing the thickness distribution on the surface of cartilages
and other helper functions

Created by zhenlinx on 6/13/18
"""
import SimpleITK as sitk
import numpy as np
import os

import time
from functools import partial
from multiprocessing import Pool
from sklearn.cluster import KMeans
import matplotlib as mpl
from skimage import measure

import pymesh
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
# from datasets import *
# from transforms import *
import glob
from numpy.linalg import norm
from misc.image_transforms import sitk_read_image


def fix_mesh(mesh, detail="normal"):
    bbox_min, bbox_max = mesh.bbox
    diag_len = norm(bbox_max - bbox_min)
    if detail == "normal":
        target_len = diag_len * 5e-3
    elif detail == "high":
        target_len = diag_len * 2.5e-3
    elif detail == "low":
        target_len = diag_len * 1e-2
    print("Target resolution: {} mm".format(target_len))

    count = 0
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)
    mesh, __ = pymesh.split_long_edges(mesh, target_len)
    num_vertices = mesh.num_vertices
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len,
                                               preserve_feature=True)
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)
        if mesh.num_vertices == num_vertices:
            break

        num_vertices = mesh.num_vertices
        print("#v: {}".format(num_vertices))
        count += 1
        if count > 10: break

    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh = pymesh.compute_outer_hull(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5)
    mesh, __ = pymesh.remove_isolated_vertices(mesh)

    return mesh


def get_largest_mesh(meshes, num_max=1, merge=True):
    """
    Get the num_max largest mesh(es) from a list of meshes
    :param meshes:
    :param num_max:
    :return: largest mesh(es): a list if num_max>1
    """
    assert (num_max > 0 and isinstance(num_max, int)), "num_max must be an >1 integer"

    num_vertices = np.array([meshes[i].num_vertices for i in range(len(meshes))])
    largest_mesh_indices = num_vertices.argsort()[-1:(-num_max - 1):-1]

    if not merge:
        return [meshes[i] for i in largest_mesh_indices]
    else:
        return pymesh.merge_meshes([meshes[i] for i in largest_mesh_indices])


def get_main_mesh(meshes, threshold=1, merge=True):
    """
    Get the main mesh(es) with a threhold of number of vertices from a list of meshes
    The goal is to remove small connected-component caused by segmentation artifacts
    :param meshes:
    :param threshold: meshes with number of vertices below the threshold will be removed
    :return: largest mesh(es): a list if num_max>1
    """
    assert (threshold > 1 and isinstance(threshold, int)), "threshold must be an >1 integer"

    num_vertices = np.array([meshes[i].num_vertices for i in range(len(meshes))])

    ordered_mesh_size_indices = num_vertices.argsort()[::-1]

    main_mesh_indices = []

    for ind in ordered_mesh_size_indices:
        if num_vertices[ind] < threshold:
            break
        else:
            main_mesh_indices.append(ind)

    if not merge:
        return [meshes[i] for i in main_mesh_indices]
    else:
        return pymesh.merge_meshes([meshes[i] for i in main_mesh_indices])

def get_neighbors(mesh, root, element, search_range=1, edge_only=False):
    """
    get neighbors of given element(vertices/faces) on mesh
    :param mesh: a PyMesh mesh object
    :param root: id of root face/vertex
    :param element: 'face' or 'f' for searching faces, and 'vertex' or 'v' for searching vertices
    :param search_range: level of searching rings on mesh
    :param edge_only: if True, only return neighbors at the edge of search range, otherwise return all neighbors within it.
    :return:
    """
    mesh.enable_connectivity()
    to_be_visited_list = []  # searching faces
    visited_set = set()  # visited neighbor faces
    # a dictionary to maintain meta information (used for path formation)
    search_dist = dict()
    search_dist[root] = 0
    to_be_visited_list.append(root)
    while not to_be_visited_list == []:
        current_node = to_be_visited_list[0]

        if search_dist[current_node] < search_range:
            # For each child of the current tree process
            temp_neighbors = None
            if element == 'face' or element == 'f':
                temp_neighbors = mesh.get_face_adjacent_faces(current_node)
            elif element == 'vertex' or element == 'v':
                temp_neighbors = mesh.get_vertex_adjacent_vertices(current_node)
            else:
                ValueError("Wrong Element Type: can only be  'v'/'vertex' for vertex, or 'f'/'face' for face")

            for neighbor in temp_neighbors:

                # The node has already been processed, so skip over it
                # if neighbor in visited_set:
                if neighbor in search_dist.keys():
                    continue

                # The child is not enqueued to be processed, so enqueue this level of children to be expanded
                if neighbor not in to_be_visited_list:
                    # create metadata for these nodes
                    if not neighbor in search_dist.keys():
                        search_dist[neighbor] = search_dist[current_node] + 1
                    else:
                        search_dist[neighbor] = min(search_dist[neighbor], search_dist[current_node] + 1)
                    # enqueue these nodes
                    to_be_visited_list.append(neighbor)

        # We finished processing the root of this subtree, so add it to the closed set
        to_be_visited_list.remove(current_node)
        if not current_node == root:
            if (not edge_only) or search_dist[current_node] == search_range:
                visited_set.add(current_node)
        pass

    return list(visited_set)


def mesh_process_pool_init(mesh):
    global MESH
    MESH = mesh


def smooth_face_label(id, face_labels, smooth_rings):
    """
    Given an id of triangle face, set its label according to the majority of its neighbors' labels
    This function is for parallel processing as all progresses share the global mesh object MESH
    :param id:
    :param face_labels:
    :param smooth_rings:
    :return:
    """
    neighbor_faces = get_neighbors(MESH, id, 'face', search_range=smooth_rings)

    if np.sum(face_labels[neighbor_faces]) < 0:
        return -1
    elif np.sum(face_labels[neighbor_faces]) > 0:
        return 1
    else:
        return face_labels[id]

def smooth_face_label_single_core(mesh, face_labels, smooth_rings):
    """
    Single process version of smooth the labels of all faces
    :param mesh:
    :param face_labels:
    :param smooth_rings:
    :return:
    """

    smoothed_labels = np.zeros(face_labels.shape)
    for id in range(len(face_labels)):
        neighbor_faces = get_neighbors(mesh, id, 'face', search_range=smooth_rings)
        if np.sum(face_labels[neighbor_faces]) < 0:
            smoothed_labels[id] = -1
        elif np.sum(face_labels[neighbor_faces]) > 0:
            smoothed_labels[id] = 1
        else:
            smoothed_labels[id] = face_labels[id]
    return smoothed_labels

def smooth_mesh_segmentation(mesh, face_labels, smooth_rings, max_rings=None, n_workers=2):
    """
    Spatially smooth the binary labels of face labels on a surface mesh, the smoothing is done by nearest neighbors
    The neighbors is controlled by smooth-rings which is defined as the discrete "geodesic" distance to the faces.
    E.g. smooth_rings=1 will use the faces directly connected to the target face as neighbors,
        smooth_rings=2 will add the faces directly connected to the ring1 neighbors as neighbors.
    if smooth_rings = 0, smoothing will not be excuted

    The smoothing will be repeated with +1 smooth_ring until it reachs max_rings or the segmentation is sufficiently
    smoothed (after segmentation, each connected component in the original mesh is segemented into two connected components).

    When max_ring is None, it will be set as smooth_ring and therefore only one smoothing iteration will be excuted.

    :param mesh:
    :param face_labels: The binary labeling have to be -1 or 1
    :param smooth_rings: size of smoothing rings(neighbor range on mesh graph)
    :param max_rings: if max_rings is given, the smooth_rings keeps growing until the smoothed mesh has no additional
    disconnected meshes or reaching maximum iterations
    :returns inner_mesh(label -1, surface touching bones), outer_mesh(label 1),
    inner_face_list(face indices of inner mesh), outer_face_list(face indices of outer mesh)
    """
    if smooth_rings == 0:
        inner_face_list = np.where(face_labels == -1)[0]
        outer_face_list = np.where(face_labels == 1)[0]

        inner_mesh = pymesh.submesh(mesh, inner_face_list, num_rings=0)
        outer_mesh = pymesh.submesh(mesh, outer_face_list, num_rings=0)
        return inner_mesh, outer_mesh, inner_face_list, outer_face_list

    if max_rings is None:
        max_rings = smooth_rings

    assert smooth_rings <= max_rings, "ERROR: Smoothing rings must be no more than max rings"

    mesh.enable_connectivity()

    while True:
        # mn test; todo maybe, remove again; use fix_mesh instead?
        # mesh, _ = pymesh.remove_duplicated_vertices(mesh)

        # one iteration of smoothing
        if n_workers>1:
            with Pool(processes=n_workers, initializer=mesh_process_pool_init, initargs=(mesh,)) as pool:
                smoothed_label = pool.map(partial(smooth_face_label, face_labels=face_labels, smooth_rings=smooth_rings),
                                          range(len(face_labels)))
        else:
            smoothed_label = smooth_face_label_single_core(mesh, face_labels, smooth_rings)
        smoothed_label = np.array(smoothed_label)

        # get faces with each label
        inner_face_list = np.where(smoothed_label == -1)[0]
        outer_face_list = np.where(smoothed_label == 1)[0]

        # get two segmented meshes
        inner_mesh = pymesh.submesh(mesh, inner_face_list, num_rings=0)
        outer_mesh = pymesh.submesh(mesh, outer_face_list, num_rings=0)

        # keep growing neighbor ring size for smoothing untill no artifacts
        num_mesh_components = len(pymesh.separate_mesh(mesh))
        if (len(pymesh.separate_mesh(inner_mesh)) == num_mesh_components and
                len(pymesh.separate_mesh(outer_mesh)) == num_mesh_components):
            print("Well smoothed mesh segmentation")
            break
        elif smooth_rings >= max_rings:
            print("Reach maximum smoothing rings")
            break
        else:
            print("Smooth kernal {} is small. Now using size {}".format(smooth_rings, smooth_rings + 1))
            smooth_rings += 1

        face_labels = smoothed_label

    return inner_mesh, outer_mesh, inner_face_list, outer_face_list


def split_femoral_cartilage_surface(mesh, smooth_rings=1, max_rings=None, n_workers=2):
    """
    Split a cartilage surface mesh into the inner and outer surface
    :param mesh:femoral cartilage surface mesh
    :param smooth_rings:
    :param max_rings
    :return: inner_mesh(label -1, surface touching bones), outer_mesh(label 1),
    inner_face_list(face indices of inner mesh), outer_face_list(face indices of outer mesh)
    """
    mesh.add_attribute("face_normal")
    mesh.add_attribute("face_centroid")

    # face normals are towards the inner of cartilage surface
    face_normal = mesh.get_attribute("face_normal").reshape([-1, 3])
    face_centroid = mesh.get_attribute("face_centroid").reshape([-1, 3])

    # get the center of bounding box of femoral cartilage
    bbox_min, bbox_max = mesh.bbox
    center = (bbox_min + bbox_max) / 2

    inner_outer_label_list = np.zeros(mesh.num_faces)  # up:1, down:-1
    for k in range(mesh.num_faces):
        # get the direction from the center to the current face centroid
        connect_direction = center - face_centroid[k, :]

        # if the direction is the same with the normal, then it is outer side (labeled as -1) of cartilage surface
        # we only cares the direction on x-y plane
        if np.dot(connect_direction[:2], face_normal[k, :2]) < 0:
            inner_outer_label_list[k] = 1
        else:
            inner_outer_label_list[k] = -1

    return smooth_mesh_segmentation(mesh, inner_outer_label_list, smooth_rings=smooth_rings, max_rings=max_rings,
                                    n_workers=n_workers)


def split_tibial_cartilage_surface(mesh, smooth_rings=1, max_rings=None, n_workers=10):
    """
    split femoral cartilage into two inner(touching the tibial bone) and outer surfaces by clustering face normals
    :param mesh: tibial cartilage surface mesh
    :param smooth_rings:
    :param max_rings:
    :return: inner_mesh(label -1, surface touching bones), outer_mesh(label 1),
    inner_face_list(face indices of inner mesh), outer_face_list(face indices of outer mesh)
    """
    mesh.enable_connectivity()

    mesh.add_attribute("face_centroid")
    mesh.add_attribute("face_normal")

    mesh_centroids = mesh.get_attribute('face_centroid').reshape(-1, 3)
    mesh_centroids_normalized = (mesh_centroids - np.mean(mesh_centroids, axis=0)) / \
                                (np.max(mesh_centroids, axis=0) - np.min(mesh_centroids, axis=0))

    mesh_normals = mesh.get_attribute('face_normal').reshape(-1, 3)

    # clustering normals
    features = np.concatenate((mesh_centroids_normalized * 2, mesh_normals * 1), axis=1)
    est = KMeans(n_clusters=2)
    # est = SpectralClustering(n_clusters=2)
    labels = est.fit(features).labels_

    # transfer 0/1 labels to -1/1 labels
    inner_outer_label_list = labels * 2 - 1

    # set inner surface which contains mostly positive normals
    if mesh_normals[inner_outer_label_list == -1, 1].mean() < 0:
        inner_outer_label_list = -inner_outer_label_list

    return smooth_mesh_segmentation(mesh, inner_outer_label_list, smooth_rings=smooth_rings, max_rings=max_rings,
                                    n_workers=n_workers)


def compute_mesh_thickness(mesh, cartilage, smooth_rings=1, max_rings=None, n_workers=10):
    """
    compute the thickness from each vertex on the cartilage to the opposite surface
    :param mesh(pymesh.mesh object):
    :param cartilage(string): 'FC' femoral cartilage or 'TC' tibial cartilage
    :return:thickness at each vertex of mesh
    """
    mesh.add_attribute("vertex_index")
    # mesh.add_attribute("vertex_normal")

    # split the cartilage into inner surface that interfacing the bone and the outer surface
    if cartilage == 'FC':
        inner_mesh, outer_mesh, inner_face_list, outer_face_list = split_femoral_cartilage_surface(mesh,
                                                                                                   smooth_rings=smooth_rings,
                                                                                                   max_rings=max_rings,
                                                                                                   n_workers=n_workers)
    elif cartilage == 'TC':
        inner_mesh, outer_mesh, inner_face_list, outer_face_list = split_tibial_cartilage_surface(mesh,
                                                                                                  smooth_rings=smooth_rings,
                                                                                                  max_rings=max_rings,
                                                                                                  n_workers=n_workers)
    else:
        ValueError("Cartilage can only be FC or TC")

    # you do not need this since inner mesh get vertex index/normal from original mesh
    # do this later, in case there was some mesh processing
    # inner_mesh.add_attribute("vertex_index")
    # inner_mesh.add_attribute("vertex_normal")
    #
    # outer_mesh.add_attribute("vertex_index")
    # outer_mesh.add_attribute("vertex_normal")

    # computer vertex distances to opposite surface
    inner_thickness = np.sqrt(pymesh.distance_to_mesh(outer_mesh, inner_mesh.vertices)[0])
    outer_thickness = np.sqrt(pymesh.distance_to_mesh(inner_mesh, outer_mesh.vertices)[0])

    # combine into a single thickness list
    thickness = np.zeros(mesh.num_vertices)
    thickness[inner_mesh.get_attribute('vertex_index').astype(int)] = inner_thickness
    thickness[outer_mesh.get_attribute('vertex_index').astype(int)] = outer_thickness
    return thickness


def get_cartilage_surface_mesh_from_segmentation_array(FC_prob, TC_prob, spacing, thickness=True,
                                                       save_path_FC=None, save_path_TC=None,
                                                       prob=True, transform=None):
    """
    Extract cartilage mesh from segmentation (3d numpy array) and compute the distance of mesh vetices to the opposite surface
    :param segmentation: HxWxD(xC), three class segmentation (probability) map of femoral cartilage, tibial cartilage
     and background
    :param spacing: spacing of segmentation map in H,W,D
    :param prob: if the input segmentation is probability map
    :param transform: the transformation that to map the voxel coordinates (default) of vertices to world coordinates
                    if None, keep the voxel coordinates (included the spacing though),
                    otherwise it should be a tuple of two numpy arrays (R, T)
                    R is a 3x3 rotation matrix and T is the translation vetor of length 3
                    The world coordinates are computed by P_w = P_v x R + T

    :return: meshes of femoral and tibial cartilage with the additional attribute "distance":

    """

    # Use marching cubes to obtain the surface mesh of shape
    print("Extract surfaces")
    FC_verts, FC_faces, FC_normals, FC_values = measure.marching_cubes_lewiner(FC_prob, 0.5,
                                                                               spacing=spacing,
                                                                               step_size=1, gradient_direction="ascent")

    TC_verts, TC_faces, TC_normals, TC_values = measure.marching_cubes_lewiner(TC_prob, 0.5,
                                                                               spacing=spacing,
                                                                               step_size=1, gradient_direction="ascent")

    FC_mesh = pymesh.form_mesh(FC_verts, FC_faces)
    TC_mesh = pymesh.form_mesh(TC_verts, TC_faces)

    FC_mesh, _ = pymesh.remove_duplicated_vertices(FC_mesh)
    TC_mesh, _ = pymesh.remove_duplicated_vertices(TC_mesh)

    FC_mesh_main = get_main_mesh(pymesh.separate_mesh(FC_mesh), threshold=1000, merge=True)
    TC_mesh_main = get_main_mesh(pymesh.separate_mesh(TC_mesh), threshold=1000, merge=True)

    if thickness:
        print("Compute FC mesh thickness")
        FC_thickness = compute_mesh_thickness(FC_mesh_main, cartilage='FC', smooth_rings=10, max_rings=None,
                                              n_workers=20)
        print('Done computing FC mesh thickness')

        print("Compute TC mesh thickness")
        TC_thickness = compute_mesh_thickness(TC_mesh_main, cartilage='TC', smooth_rings=10, max_rings=None,
                                              n_workers=20)
        print('Done computing TC mesh thickness')

    if transform:
        FC_mesh_main = pymesh.form_mesh(voxel_to_world_coord(FC_mesh_main.vertices, transform),
                                        FC_mesh_main.faces)
        TC_mesh_main = pymesh.form_mesh(voxel_to_world_coord(TC_mesh_main.vertices, transform),
                                        TC_mesh_main.faces)

    if thickness:
        FC_mesh_main.add_attribute("vertex_thickness")
        FC_mesh_main.set_attribute("vertex_thickness", FC_thickness)

        TC_mesh_main.add_attribute("vertex_thickness")
        TC_mesh_main.set_attribute("vertex_thickness", TC_thickness)

        if save_path_FC:
            pymesh.save_mesh(save_path_FC, FC_mesh_main, "vertex_thickness", ascii=True)
        if save_path_TC:
            pymesh.save_mesh(save_path_TC, TC_mesh_main, "vertex_thickness", ascii=True)
    else:
        if save_path_FC:
            pymesh.save_mesh(save_path_FC, FC_mesh_main, ascii=True)
        if save_path_TC:
            pymesh.save_mesh(save_path_TC, TC_mesh_main, ascii=True)

    return FC_mesh_main, TC_mesh_main


def get_cartilage_surface_mesh_from_segmentation_file(segmentation_file, thickness=True,
                                                      save_path_FC=None, save_path_TC=None,
                                                      prob=True, coord='voxel'):
    """
    compute cartilage thickness from a segmentation file
    :param segmentation_file: the image file and a tupe of seperated files of the FC/TC segmentation mask/probmaps
    :param save_path_FC:
    :param save_path_TC:
    :param prob: if True, the input segmentation is probability maps, otherwise is segmentation mask
    :param coord: the coordinate system the output mesh lie in.
                  'voxel': the image space;
                  'nifti': the world space follows convention in nifti definition, used in ITK-SNAP and NiftyReg
                  'itk': the world space follows ITK convention, used in ITK and simpleITK
    :return:
    """
    if type(segmentation_file) == str:
        segmentation = sitk_read_image(segmentation_file)

        # the np array from itk are ordered in z,y,x
        segmentation_np = np.swapaxes(sitk.GetArrayFromImage(segmentation), 0, 2)

        if prob:
            FC_prob = segmentation_np[:, :, :, 0]
            TC_prob = segmentation_np[:, :, :, 1]
        else:
            FC_prob = (segmentation == 1).astype(float)
            TC_prob = (segmentation == 2).astype(float)

        # get rotation matrix and translation vector for mapping from voxel space to world space
        if coord == 'voxel':
            transform = None
        elif coord == 'nifti':
            transform = get_voxel_to_world_transform_nifti(segmentation[0])

        spacing = segmentation.GetSpacing()

    elif type(segmentation_file) == tuple:
        if type(segmentation_file[0]) == str and type(segmentation_file[1]) == str:
            segmentation = [sitk_read_image(file) for file in segmentation_file]
            FC_prob = np.swapaxes(sitk.GetArrayFromImage(segmentation[0]), 0, 2).astype(float)
            TC_prob = np.swapaxes(sitk.GetArrayFromImage(segmentation[1]), 0, 2).astype(float)

            # the np array from itk are ordered in z,y,x
            if coord == 'voxel':
                transform = None
            elif coord == 'nifti':
                transform = get_voxel_to_world_transform_nifti(segmentation[0])
            spacing = segmentation[0].GetSpacing()
        else:
            TypeError("The segmentation files must be a tuple of strings, but a tuple of ({}, {}) is given".format(
                type(segmentation_file[0]), type(segmentation_file[1])))

    else:
        TypeError("The segmentation file must be a str type or a tuple of strings, but {} is given".format(type(segmentation_file)))


    return get_cartilage_surface_mesh_from_segmentation_array(FC_prob, TC_prob,
                                                              spacing=spacing,
                                                              thickness=thickness,
                                                              save_path_TC=save_path_TC,
                                                              save_path_FC=save_path_FC,
                                                              prob=prob,
                                                              transform=transform)


def voxel_to_world_coord(input_points, transform):
    """
    A function that transfer the voxel coordinates to world coordinates
    :param input_points: Nx3 array, the points of coordinates in the voxel space
    :param transform: a tuple of two numpy arrays (R, T) represents the transformation that to map the voxel coordinates
            of vertices to world coordinates, where R is a 3x3 rotation matrix and T is the translation vetor of length 3
            The world coordinates are computed by P_w = R x P_v  + T
    :return: output_points: Nx3 array, the points of coordinates in the world space

    """
    assert input_points.shape[1] == 3 and len(input_points.shape) == 2
    R, T = transform
    return R.dot(input_points.transpose()).transpose() + T



def world_to_voxel_coord(input_points, transform):
    """
    A function that transfer the world coordinates to voxel coordinates
    :param input_points: Nx3 array, the points of coordinates in the voxel space
    :param transform: a tuple of two numpy arrays (R, T) represents the transformation that to map the voxel coordinates
            of vertices to world coordinates, where R is a 3x3 rotation matrix and T is the translation vetor of length 3
            The world coordinates are computed by P_w = R x P_v  + T
    :return: output_points: Nx3 array, the points of coordinates in the voxel space

    """
    assert input_points.shape[1] == 3 and len(input_points.shape) == 2
    R, T = transform
    return ((np.linalg.inv(R)).dot((input_points-T).transpose())).transpose()


def get_voxel_to_world_transform_nifti(reference_image):
    """
    Get the Rotation matrix and the translation vector for transfrom from voxel coordinates to world coordinates

    See the nifti convention https://github.com/SuperElastix/niftyreg/blob/62af1ca6777379316669b6934889c19863eaa708/reg-io/nifti/nifti1.h#L977

    :param reference_image: the image which defines the voxel-world transform
    :return:
    """
    if type(reference_image) is str:
        reference_image = sitk_read_image(reference_image)

    b, c, d = float(reference_image.GetMetaData('quatern_b')), float(reference_image.GetMetaData('quatern_c')), \
              float(reference_image.GetMetaData('quatern_d'))
    a = np.sqrt(1.0 - (b * b + c * c + d * d))
    R = np.array([[a * a + b * b - c * c - d * d, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
                  [2 * b * c + 2 * a * d, a * a + c * c - b * b - d * d, 2 * c * d - 2 * a * b],
                  [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a * a + d * d - c * c - b * b]])
    T = np.array([float(reference_image.GetMetaData('qoffset_x')), float(reference_image.GetMetaData('qoffset_y')),
                  float(reference_image.GetMetaData('qoffset_z'))])
    return R, T


def save_points(points, file):
    with open(file, 'w') as f:
        for n in range(points.shape[0]):
            f.write("{} {} {}\n".format(points[n, 0], points[n, 1], points[n, 2]))


def read_points(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        n = len(lines)
        points = np.zeros([n, 3])
        for k, line in enumerate(lines):
            p = line.split(' ')
            points[k, :] = list(map(float, p[:3]))

    return points


def get_points_from_mesh(input_mesh, output_points_file=None, transform=None):
    """
    Get vertices coordinates as text tile in lines of "x y z",
    and save into files if the filename is given
    :param input_mesh: mesh object or a mesh filename
    :param output_points_file:
    :return: point cloud of mesh vertices after transform
    """
    if type(input_mesh) == str:
        input_mesh = pymesh.load_mesh(input_mesh)
    if transform:
        points = voxel_to_world_coord(input_mesh.vertices, transform)
    else:
        points = input_mesh.vertices

    if type(output_points_file) == str:
        save_points(points, output_points_file)

    return points


def mesh_to_nifti_world_coord(mesh_file, reference_image, output_points_file=None, inWorld=False):
    """
    Transfer the vertices coordinates of a mesh (defined in image space) into the world coordinate system

    :param mesh_file: the mesh file where coordinates of vertices are in voxel space
    :param reference_image: the image which defines the voxel-world transform
    :param output_points_file(optional): a txt file that stores the coordinates that were transfered to world space,
                                        if none do nothing.
    :return: point cloud of mesh vertices transformed to world coordinates
    """
    if inWorld:
        transform = None
    else:
        transform = get_voxel_to_world_transform_nifti(reference_image)
    points = get_points_from_mesh(mesh_file, output_points_file, transform)
    return points


def modify_mesh_with_new_vertices(mesh, points, output_mesh_file=None):
    """
    change the vertices coordinates using the points coordinates stored in a txt file
    :param mesh: input mesh whose vertices will be replaced. (mesh object or mesh filename)
    :param points: input points of the new vertices. (a Nx3 numpy array or a text file with lines of 'x y z')
        The number of points has to be the same as the number of mesh vertices.
    :param output_mesh_file (optional): saved updated mesh file
    :return: new_mesh: mesh with updated vertices
    """

    if type(mesh) is str:
        mesh = pymesh.load_mesh(mesh)
    if type(points) is str:
        points = read_points(points)
    new_mesh = pymesh.form_mesh(points, mesh.faces)
    attribute_to_save = []
    for attribute in mesh.attribute_names:
        if not attribute in ['face_vertex_indices', 'vertex_x', 'vertex_y', 'vertex_z']:
            new_mesh.add_attribute(attribute)
            new_mesh.set_attribute(attribute, mesh.get_attribute(attribute))
            attribute_to_save.append(attribute)

    if output_mesh_file:
        pymesh.save_mesh(output_mesh_file, new_mesh, *attribute_to_save, ascii=True)

    return new_mesh


def map_thickness_to_atlas_mesh(atlas_mesh, source_mesh, atlas_mapped_file=None):
    """
    Map thickness of a registered mesh to the atlas mesh
    :param atlas_mesh: atlas mesh which the source mesh was registered to
    :param source_mesh: the mesh with thickness that has been registered to atlas space
    :param atlas_mapped_file: the file to save the atlas mesh with mapped thickness
    :return: atlas mesh with mapped thickness
    """
    if type(atlas_mesh) == str:
        atlas_mesh = pymesh.load_mesh(atlas_mesh)
    elif isinstance(atlas_mesh) == pymesh.Mesh:
        atlas_mesh = atlas_mesh.copy()
    else:
        TypeError("atlas mesh is either a mesh file or a pymesh.mesh ")

    if type(source_mesh) == str:
        source_mesh = pymesh.load_mesh(source_mesh)

    map_vertex_attribute(source_mesh, atlas_mesh, 'vertex_thickness', threshold=6)
    if type(atlas_mapped_file) == str:
        pymesh.save_mesh(atlas_mapped_file, atlas_mesh, 'vertex_thickness', ascii=True)

    return atlas_mesh


def map_vertex_attribute(mesh1, mesh2, attr_name, threshold, bvh=None):
    """ Map vertex attribute from mesh1 to mesh2 based on closest points.
    Modified from code in Pymesh map_attributes.py

    Args:
        mesh1 (:class:`Mesh`): Source mesh, where the attribute is defined.
        mesh2 (:class:`Mesh`): Target mesh, where the attribute is mapped to.
        attr_name (``string``): Attribute name.
        threshold (``int``): Any vertex with closest distance above the thrshold is set as zero
        bvh (:class:`BVH`): Pre-computed Bounded volume hierarchy if available.

    A new attribute with name ``attr_name`` is added to ``mesh2``.
    """

    import numpy as np
    from pymesh.aabb_tree import BVH

    assert (mesh1.dim == mesh2.dim);
    assert (mesh1.vertex_per_face == 3);
    assert (mesh2.vertex_per_face == 3);
    assert (mesh1.has_attribute(attr_name));
    values = mesh1.get_vertex_attribute(attr_name);

    if bvh is None:
        bvh = BVH(dim=mesh1.dim);
        bvh.load_mesh(mesh1);
    dists, closest_faces, p = bvh.lookup(mesh2.vertices);

    vertices = mesh1.vertices;
    faces = mesh1.faces;
    v0 = vertices[faces[closest_faces, 0], :];
    v1 = vertices[faces[closest_faces, 1], :];
    v2 = vertices[faces[closest_faces, 2], :];

    a01 = np.linalg.norm(np.cross(v0 - p, v1 - p), axis=1).reshape((-1, 1));
    a12 = np.linalg.norm(np.cross(v1 - p, v2 - p), axis=1).reshape((-1, 1));
    a20 = np.linalg.norm(np.cross(v2 - p, v0 - p), axis=1).reshape((-1, 1));

    a = a01 + a12 + a20;

    val_0 = values[faces[closest_faces, 0], :];
    val_1 = values[faces[closest_faces, 1], :];
    val_2 = values[faces[closest_faces, 2], :];

    target_val = (val_0 * a12 + val_1 * a20 + val_2 * a01) / a;

    target_val[dists>threshold] = 0

    if not mesh2.has_attribute(attr_name):
        mesh2.add_attribute(attr_name);
    mesh2.set_attribute(attr_name, target_val);


def __generate_mask(vertice, mask, interx, intery, startx, starty):
    import math
    x, y = zip(*vertice)
    for i in range(len(vertice)):
        idx = math.floor((x[i] - startx) / interx)
        idy = math.floor((y[i] - starty) / intery)
        mask[idy, idx] = False
        for j in range(idx - 1, idx + 2):
            for k in range(idy - 1, idy + 2):
                mask[k, j] = False
    return mask

def __map_thickness_to_2D_projection(embedded, thickness, ninter=100, min_thickness=-1., fpth_np=None, fpth_png=None, name=''):
    """

    :param embedded: numpy array, 2d embedded atlas mesh
    :param thickness: numpy array, thickness
    :param ninter: mask resolution
    :param min_thickness: the threshold on minumum of thickness
    :param fpth_np: saving path of numpy array of 2d thickness map
    :param fpth_png: saving path of 2d project image
    :param name: projection name
    :return:
    """

    xmin, xmax, ymin, ymax = min(embedded[:, 0]), max(embedded[:, 0]), min(embedded[:, 1]), max(embedded[:, 1])
    rangex = xmax - xmin
    rangey = ymax - ymin
    interx = rangex / ninter
    intery = rangey / ninter
    xi = np.arange(xmin - interx * 5, xmax + interx * 5, interx)
    yi = np.arange(ymin - intery * 5, ymax + intery * 5, intery)
    xi, yi = np.meshgrid(xi, yi)
    mask = np.zeros_like(xi) == 0
    mask = __generate_mask(embedded, mask, interx, intery, xi.min(), yi.min())
    x, y = zip(*embedded)
    if min_thickness>0:
        thickness[thickness<min_thickness]=min_thickness
    z = thickness
    zi = griddata((x, y), z, (xi, yi), method='linear')
    #zi[mask] = 100
    zi[mask] = np.nan

    # zi now has the thickness, we can write this out as a numpy file
    np.save(file=fpth_np,arr=zi)

    contour_num = 80
    maxz = max(z)
    plt.contourf(xi, yi, zi, np.arange(0.0, maxz + maxz / contour_num, maxz / contour_num))
    plt.axis('equal')
    # plt.xlabel('xi', fontsize=16)
    # plt.ylabel('yi', fontsize=16)
    plt.axis('off')
    plt.colorbar().ax.tick_params(labelsize=10)
    font = {'size': 10}
    plt.title(name, font)
    if fpth_png is not None:
        plt.savefig(fpth_png,dpi=300)
        plt.close('all')
    else:
        plt.show()
        plt.clf()


def map_thickness_to_2D_projection(atlas_mesh_with_thickness, atlas_2d_map_file=None, map_2d_base_filename=None, name='', overwrite=False):
    """
    Map thickness of a registered mesh to the 2d projection of the atlas mesh
    :param atlas_mesh_with_thickness: atlas mesh which the source mesh was registered to
    :param map_2d_base_filename: filename to save the 2d contour with mapped thickness (as png) and the raw values as a numpy array
    :param  name: name of the projection
    :param  overwrite: overwrite if the files already exist
    :return:
    """

    map_2d_file_np = map_2d_base_filename # numpy will automatically add .npy as suffix
    map_2d_file_png = map_2d_base_filename + '.png'

    if overwrite is False:
        if os.path.exists(map_2d_file_np) and os.path.exists(map_2d_file_png):
            print('Thickness 2D projection, not recomputing as {} and {} exist.'.format(map_2d_file_np,map_2d_file_png))
            return

    if type(atlas_mesh_with_thickness) == str:
        atlas_mesh_with_thickness = pymesh.load_mesh(atlas_mesh_with_thickness)
    elif isinstance(atlas_mesh_with_thickness) == pymesh.Mesh:
        atlas_mesh_with_thickness = atlas_mesh_with_thickness.copy()
    else:
        TypeError("atlas mesh is either a mesh file or a pymesh.mesh ")

    # if type(source_mesh) == str:
    #     source_mesh = pymesh.load_mesh(source_mesh)

    # first : map_thickness_to_atlas_mesh
    # map_vertex_attribute(source_mesh, atlas_mesh, 'vertex_thickness', threshold=3)
    # second: load atlas 2d map
    embedded = np.load(atlas_2d_map_file)
    # third:  do projection
    thickness = atlas_mesh_with_thickness.get_attribute("vertex_thickness")
    thickness = thickness.copy()
    print(np.where(thickness==0)[0].shape)

    __map_thickness_to_2D_projection(embedded, thickness, ninter=300, min_thickness=-1,fpth_np=map_2d_file_np,fpth_png=map_2d_file_png, name=name)

def surface_distance(source_mesh, target_mesh):
    """
    Measure the surface distance from the source mesh to the target mesh
    :param source_mesh:
    :param target_mesh:
    :return:
    """
    if type(source_mesh) == str:
        source_mesh = pymesh.load_mesh(source_mesh)
    if type(target_mesh) == str:
        target_mesh = pymesh.load_mesh(target_mesh)

    distances_sqr, _, _ = pymesh.distance_to_mesh(target_mesh, source_mesh.vertices)
    distances = np.sqrt(distances_sqr)

    return np.max(distances), np.min(distances), np.median(distances), np.percentile(distances, 95)


def test_mesh_points():

    import pymesh
    from cartilage_shape_processing import get_voxel_to_world_transform_nifti, voxel_to_world_coord
    ref_image = '/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/atlas_60_LEFT_baseline_NMI/atlas_step_10.nii.gz'
    mesh_file = 'test_analysis/atlas_TC_mesh.ply'
    points_world_file = 'test_analysis/atlas_TC_points.txt'
    mesh_world = modify_mesh_with_new_vertices(mesh_file, points_world_file, 'atlas_TC_mesh.ply')
    pass


def plot_mesh_segmentation(mesh1, mesh2):
    # mesh1, mesh2, _, _ = split_femoral_cartilage_surface(mesh, smooth_rings=10,
    #                                                      max_rings=None, n_workers=20)
    import visvis as vv

    app = vv.use()
    a1 = vv.subplot(111)
    FC_vis_up = vv.mesh(mesh1.vertices, mesh1.faces)
    # FC_vis.colormap = vv.CM_JET
    FC_vis_up.faceColor = 'r'
    FC_vis_down = vv.mesh(mesh2.vertices, mesh2.faces)
    FC_vis_down.faceColor = 'b'
    app.Run()


def main():
    import visvis as vv

    # read a cartilage
    results_dir = "/playpen-raid/zhenlinx/Code/OAI_segmentation/unet_3d/results/ACRNN_withPreModelNifti_corrected_cropped_rescaled_patch_128_128_32_batch_1_sample_0.01-0.02_lr_0.0001_01022018_183359"
    cartilage_prop_filenames = glob.glob(os.path.join(results_dir, "*prediction_step3_16_reflect_probmap.nii.gz"))
    # cartilage_prop_filename = "9085290_20040915_SAG_3D_DESS_LEFT_016610243503_prediction_step3_16_reflect_probmap.nii.gz"
    # label_map_filename = "9085290_20040915_SAG_3D_DESS_LEFT_016610243503_prediction_step3_16_reflect.nii.gz"
    for cartilage_prop_filename in cartilage_prop_filenames[:1]:
        cartilage_prop = sitk_read_image(cartilage_prop_filename, results_dir)
        # label_map = read_image(label_map_filename, results_dir)

        # get numpy array
        cartilage_prop_np = sitk.GetArrayFromImage(cartilage_prop)

        FC_mesh_main, TC_mesh_main = get_cartilage_surface_mesh_from_segmentation_array(cartilage_prop_np[:, :, :, 0],
                                                                                        cartilage_prop_np[:, :, :, 1],
                                                                                        cartilage_prop.GetSpacing()[
                                                                                        ::-1]
                                                                                        )

        app = vv.use()
        a1 = vv.subplot(311)
        FC_vis = vv.mesh((FC_mesh_main.vertices), FC_mesh_main.faces, values=FC_mesh_main.get_attribute('vertex_thickness'))
        TC_vis = vv.mesh((TC_mesh_main.vertices), TC_mesh_main.faces, values=TC_mesh_main.get_attribute('vertex_thickness'))
        FC_vis.colormap = vv.CM_JET
        TC_vis.colormap = vv.CM_JET
        vv.colorbar()
        a2 = vv.subplot(312)
        FC_vis_copy = vv.mesh((FC_mesh_main.vertices), FC_mesh_main.faces,
                              values=FC_mesh_main.get_attribute('vertex_thickness'))
        FC_vis_copy.colormap = vv.CM_JET
        vv.colorbar()
        a3 = vv.subplot(313)
        TC_vis_copy = vv.mesh((TC_mesh_main.vertices), TC_mesh_main.faces,
                              values=TC_mesh_main.get_attribute('vertex_thickness'))
        TC_vis_copy.colormap = vv.CM_JET
        vv.colorbar()
        app.Run()


if __name__ == '__main__':
    # test_mesh_points()
    main()
    # mesh = pymesh.load_mesh('./test_analysis/9007827_20041006_SAG_3D_DESS_LEFT_016610263603_mesh_FC.ply')
    # mesh = pymesh.load_mesh('./test_analysis/atlas_FC_mesh_world.ply')
    # mesh1, mesh2, _, _ = split_femoral_cartilage_surface(mesh, smooth_rings=10,
    #                                                      max_rings=None, n_workers=20)
    #
    # mesh = pymesh.load_mesh('./test_analysis/atlas_TC_mesh_world.ply')
    # mesh1, mesh2, _, _ = split_tibial_cartilage_surface(mesh, smooth_rings=10,
    #                                                     max_rings=None, n_workers=20)
    # plot_mesh_segmentation(mesh1, mesh2)
