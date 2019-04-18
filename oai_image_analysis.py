#!/usr/bin/env python
"""
Created by zhenlinxu on 01/04/2019
"""
import os
import sys

sys.path.append(os.path.realpath(".."))
from multiprocessing import Pool
from functools import partial

import SimpleITK as sitk
import numpy as np

from segmentation.segmenter import Segmenter3DInPatchClassWise
from registration.registers import NiftyReg, AVSMReg
from shape_analysis.cartilage_shape_processing import get_cartilage_surface_mesh_from_segmentation_file, \
    map_thickness_to_atlas_mesh, surface_distance, map_thickness_to_2D_projection
from data.pre_process import label2image, image_normalize, reset_sitk_image_coordinates, flip_left_right

class OAIImageAnalysis:
    def __init__(self,use_nifti=True):
        self.segmenter = Segmenter3DInPatchClassWise()
        self.use_nifti = use_nifti
        self.register = NiftyReg() if use_nifti else AVSMReg()

        self.atlas_image_file = None
        # self.atlas_mask_file = None
        self.atlas_FC_mesh_file = None
        self.atlas_TC_mesh_file = None

        self.atlas_FC_2D_map_file = None
        self.atlas_TC_2D_map_file = None

        self.surface_distance_FC = []  # should be Nx4 array, columns are max, min, median, 95%max
        self.surface_distance_TC = []

    def set_atlas(self, atlas_image_file, atlas_FC_mesh_file, atlas_TC_mesh_file):
        self.atlas_image_file = atlas_image_file
        # self.atlas_mask_file = None
        self.atlas_FC_mesh_file = atlas_FC_mesh_file
        self.atlas_TC_mesh_file = atlas_TC_mesh_file


    def compute_atlas_2D_map(self,n_jobs=-1):
        from sklearn.manifold import MDS
        import pymesh
        mds = MDS(2, max_iter=20000, n_init=10, n_jobs=n_jobs)
        altas_2D_file_list = [self.atlas_FC_2D_map_file,self.atlas_TC_2D_map_file]
        altas_file_list = [self.atlas_FC_mesh_file,self.atlas_TC_mesh_file]
        for i, altas_2D_file in enumerate(altas_2D_file_list):
            if not os.path.isfile(altas_2D_file):
                os.makedirs(os.path.split(altas_2D_file)[1],exist_ok=True)
                print("the {} is not exist, now compute the 2d map from atlas, it will take hours, if the default setting out of memory, you may need to use"
                      "less processers, change 'n_jobs' in MDS ".format(altas_2D_file))
                mesh = pymesh.load_mesh(altas_file_list[i])
                vertices = mesh.vertices
                embedded = mds.fit_transform(vertices)
                np.save(altas_2D_file, embedded)
                print("complete, the 2D map is saved into {}".format(altas_2D_file))

    def set_atlas_2D_map(self,atlas_FC_2D_map_file='./atlas/FC_inner_embedded.npy', atlas_TC_2D_map_file='./atlas/TC_inner_embedded.npy'):
        self.atlas_FC_2D_map_file = atlas_FC_2D_map_file
        self.atlas_TC_2D_map_file =atlas_TC_2D_map_file


    def set_segmenter(self, segmenter):
        self.segmenter = segmenter

    def close_segmenter(self):
        """deinit the segmenter to release resources"""
        self.segmenter = None

    def set_register(self, register, affine_config, bspline_config):
        self.register = register
        self.affine_config = affine_config
        self.bspline_config = bspline_config

    def set_preprocess(self, bias_correct=False, reset_coord=True, normalize_intensity=True,
                       flip_to="LEFT"):
        """
        config preprocessing
        :param bias_correct: if do bias field correction
        :param reset_coord: if reset image origin and orientation
        :param normalize_intensity: if normalize image intensity into [0.1]
        :param flip_to: if flip all image to a constant side e.g. 'left' means flip all left image to right
        :return:
        """
        self.bias_correct = bias_correct
        self.flip_to = flip_to
        self.reset_coord = reset_coord
        self.normalize_intensity = normalize_intensity

    def preprocess(self, oai_image, overwrite=False):
        """

        :param oai_image: OAIImage object
        :param overwrite: if overwrite existing image files
        :return:
        """
        # oai_image.folder = os.path.join(self.save_root, str(oai_image.patient_id), oai_image.modality, oai_image.part,
        #                             self.visit_description[oai_image.visit_month])

        if not os.path.exists(oai_image.folder):
            os.makedirs(oai_image.folder, exist_ok=True)

        # oai_image.preprocessed_image_file = os.path.join(oai_image.folder, 'image_preprocessed.nii.gz')
        if overwrite or (not os.path.isfile(oai_image.preprocessed_image_file)):
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(oai_image.raw_folder)

            print("Read {} DICOM slices in {}".format(len(dicom_names), oai_image.raw_folder))
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            image = sitk.Cast(image, sitk.sitkFloat32)

            # bias field correction
            if self.bias_correct:
                # if want to use all voxels
                all_mask = label2image(np.ones(sitk.GetArrayFromImage(image).shape).astype(int),image)

                image = sitk.Add(image, 1)
                image = sitk.N4BiasFieldCorrection(image, maskImage=all_mask)
                image = sitk.Subtract(image, 1)

            # rescale the intensity
            if self.normalize_intensity:
                print("Normalizing: {}".format(oai_image.name))
                image = image_normalize(image, 0.1, 99.9, 0, 1)

            # reset original and orientation
            if self.reset_coord:
                # reset original and orientation
                print("reset coordinates: {}".format(oai_image.name))
                reset_sitk_image_coordinates(image, [0, 0, 0], [0, 0, -1, 1, 0, 0, 0, -1, 0])

            # flip the left and right:
            if self.flip_to and (self.flip_to not in oai_image.part):
                print("flipping {}".format(oai_image.name))
                image = flip_left_right(image)

            print("Saving:{} at {}".format(oai_image.name, oai_image.folder))
            sitk.WriteImage(image, oai_image.preprocessed_image_file)

    def segment_image_and_save_results(self, oai_image, overwrite=False):
        """
        Segment image and save the integer mask and probmap.
        The results are saved into a folder named by the segmenter's name under the path of image file.
        :param oai_image
        :param overwrite: if overwrite existing segmentations
        :return: None
        """
        # generate segmentation folder and files path
        # FC_probmap_file = os.path.join(oai_image.folder, 'FC_probmap.nii.gz')
        # TC_probmap_file = os.path.join(oai_image.folder, 'TC_probmap.nii.gz')

        if (not overwrite) and os.path.isfile(oai_image.FC_probmap_file) and os.path.isfile(oai_image.TC_probmap_file):
            print("Segmentations found at {}".format(oai_image.folder))
        else:
            print("Segmenting {}".format(oai_image.preprocessed_image_file))
            image = sitk.ReadImage(oai_image.preprocessed_image_file)
            FC_probmap, TC_probmap = self.segmenter.segment(image, if_output_prob_map=True, if_output_itk=True)

            sitk.WriteImage(FC_probmap, oai_image.FC_probmap_file)
            sitk.WriteImage(TC_probmap, oai_image.TC_probmap_file)


    def extract_surface_mesh(self, oai_image, overwrite=False,coord='nifti'):
        """Extract surface mesh of cartilages with thickness measurements"""

        # oai_image.FC_mesh_file = os.path.join(oai_image.folder, "FC_mesh_world.ply")
        # oai_image.TC_mesh_file = os.path.join(oai_image.folder, "TC_mesh_world.ply")

        if (not overwrite) and os.path.isfile(oai_image.FC_mesh_file) and os.path.isfile(oai_image.TC_mesh_file):
            print("{} and {} exits".format(oai_image.FC_mesh_file, oai_image.TC_mesh_file))
        else:
            get_cartilage_surface_mesh_from_segmentation_file((oai_image.FC_probmap_file, oai_image.TC_probmap_file),
                                                              save_path_FC=oai_image.FC_mesh_file,
                                                              save_path_TC=oai_image.TC_mesh_file,
                                                              thickness=True, prob=True, coord=coord,
                                                              )
        pass

    def register_image_to_atlas(self,oai_image,overwrite=False,gpu_id=0):
        if self.use_nifti:
            self.register_image_to_atlas_NiftyReg(oai_image,overwrite)
        else:
            self.register_image_to_atlas_AVSM(oai_image,overwrite,gpu_id=gpu_id)


    def register_image_to_atlas_AVSM(self,oai_image,overwrite=False, gpu_id=0):
        if overwrite:
            self.register.register_image(self.atlas_image_file, oai_image.preprocessed_image_file,
                                         lmoving_path=None, ltarget_path=None,
                                         gpu_id=gpu_id,oai_image=oai_image)


    def register_image_to_atlas_NiftyReg(self, oai_image, overwrite=False):
        """
        Register an oai_image to the atlas
        :param oai_image: (OAIImage object)
        :param eval: if evaluate the surface distance between the warped mesh and the atlas mesh
        :param overwrite:
        :return:
        """
        # oai_image.affine_transform_file = os.path.join(oai_image.folder, "affine_transform_to_atlas.txt")
        # oai_image.bspline_transform_file = os.path.join(oai_image.folder, "bspline_control_points_to_atlas.nii.gz")

        if overwrite or (not os.path.isfile(oai_image.affine_transform_file)):
            self.register.register_affine(self.atlas_image_file,
                                          oai_image.preprocessed_image_file,
                                          out_affine_file=oai_image.affine_transform_file,
                                          **self.affine_config)

        if overwrite or (not os.path.isfile(oai_image.bspline_transform_file)):
            self.register.register_bspline(self.atlas_image_file, oai_image.preprocessed_image_file,
                                           init_affine_file=oai_image.affine_transform_file,
                                           output_control_point=oai_image.bspline_transform_file,
                                           **self.bspline_config)

    def warp_mesh(self, oai_image, overwrite=False):

        # oai_image.warped_FC_mesh = os.path.join(oai_image.folder, "FC_mesh_world_to_atlas.ply")
        # oai_image.warped_TC_mesh = os.path.join(oai_image.folder, "TC_mesh_world_to_atlas.ply")
        # oai_image.inv_transform_to_atlas = os.path.join(oai_image.folder, "inv_transform_to_atlas.nii.gz")

        # generate inverse registration map
        if overwrite or (not os.path.isfile(oai_image.inv_transform_to_atlas)):
            self.register.invert_nonrigid(non_rigid_transform=oai_image.bspline_transform_file,
                                          reference_image=self.atlas_image_file,
                                          moving_image=oai_image.preprocessed_image_file,
                                          inverted_transform=oai_image.inv_transform_to_atlas)

        # warp meshes using the inverse map
        if overwrite or (not os.path.isfile(oai_image.warped_FC_mesh_file)):
            self.register.warp_mesh(oai_image.FC_mesh_file,
                                    oai_image.inv_transform_to_atlas,
                                    oai_image.preprocessed_image_file,
                                    oai_image.warped_FC_mesh_file,
                                    inWorld=True)
        if overwrite or (not os.path.isfile(oai_image.warped_TC_mesh_file)):
            self.register.warp_mesh(oai_image.TC_mesh_file,
                                    oai_image.inv_transform_to_atlas,
                                    oai_image.preprocessed_image_file,
                                    oai_image.warped_TC_mesh_file,
                                    inWorld=True)

    def eval_registration_surface_distance(self, oai_image):
        # messure the surface distance between the warped mesh and the atlas mesh
        dist_max_FC, dist_min_FC, dist_median_FC, dist_95p_FC = surface_distance(oai_image.warped_FC_mesh_file, self.atlas_FC_mesh_file)
        dist_max_TC, dist_min_TC, dist_median_TC, dist_95p_TC = surface_distance(oai_image.warped_TC_mesh_file, self.atlas_TC_mesh_file)
        print("{} distance eval(mm):".format(oai_image.name))
        print("FC distance: max:{:.4f}, min:{:.4f}, median:{:.4f}, 95 percent {:.4f}".format(dist_max_FC, dist_min_FC, dist_median_FC,
                                                                         dist_95p_FC))
        print("TC distance: max:{:.4f}, min:{:.4f}, median:{:.4f}, 95 percent {:.4f}".format(dist_max_TC, dist_min_TC, dist_median_TC,
                                                                            dist_95p_TC))
        self.surface_distance_FC.append(np.array([[dist_max_FC, dist_min_FC,dist_median_FC, dist_95p_FC]]))
        self.surface_distance_TC.append(np.array([[dist_max_FC, dist_min_FC,dist_median_FC, dist_95p_FC]]))

    def get_surface_distances_eval(self):
        FC_distances = np.vstack(self.surface_distance_FC)
        TC_distances = np.vstack(self.surface_distance_FC)
        FC_means = FC_distances.mean(axis=0)
        FC_std = FC_distances.std(axis=0)
        TC_means = TC_distances.mean(axis=0)
        TC_std = TC_distances.std(axis=0)
        print("FC distance(mm): max:{:.4f} +/- {:.4f}, min:{:.4f} +/- {:.4f}, median:{:.4f} +/- {:.4f}, 95 percent {} +/- {}".format(
            *[metric[i] for i in range(4) for metric in [FC_means, FC_std]]))
        print("TC distance(mm): max:{:.4f} +/- {:.4f}, min:{:.4f} +/- {:.4f}, median:{:.4f} +/- {:.4f}, 95 percent {} +/- {}".format(
            *[metric[i] for i in range(4) for metric in [TC_means, TC_std]]))

    def project_thickness_to_atlas(self, oai_image, overwrite=False):
        """
        use the inverse registration map to warp the oai_image mesh to atlas space
        and project the thickness to the atlas mesh
        """
        # setup paths of files to be generated
        # oai_image.FC_thickness_mapped_to_atlas_mesh = os.path.join(oai_image.folder, "atlas_FC_mesh_with_thickness.ply")
        # oai_image.TC_thickness_mapped_to_atlas_mesh = os.path.join(oai_image.folder, "atlas_TC_mesh_with_thickness.ply")

        # map the thickness on the warped meshes to the atlas meshes
        map_thickness_to_atlas_mesh(self.atlas_FC_mesh_file, oai_image.warped_FC_mesh_file, oai_image.FC_thickness_mapped_to_atlas_mesh)
        map_thickness_to_atlas_mesh(self.atlas_TC_mesh_file, oai_image.warped_TC_mesh_file, oai_image.TC_thickness_mapped_to_atlas_mesh)

    def project_thickness_to_2D(self, oai_image, overwrite=False):
        """
        TODO: implement the method to project thickness from 3D mesh to 2D grid,
         and save the file to oai_image.FC_2D_thickness_grid and oai_image.TC_2D_thickness_grid
        """
        map_thickness_to_2D_projection(self.atlas_FC_mesh_file, oai_image.warped_FC_mesh_file,self.atlas_FC_2D_map_file, oai_image.FC_2D_thickness_grid,name='FC_2D_map')
        map_thickness_to_2D_projection(self.atlas_TC_mesh_file, oai_image.warped_TC_mesh_file,self.atlas_TC_2D_map_file, oai_image.TC_2D_thickness_grid,name='TC_2D_map')

    def preprocess_parallel(self, image_list, n_workers=10, overwrite=False):
        with Pool(processes=n_workers) as pool:
            res = pool.map(partial(self.preprocess, overwrite=overwrite), image_list)


    def analyze_single_image(self, image, overwrite=False):
        self.preprocess(image, overwrite=overwrite)
        self.segment_image_and_save_results(image, overwrite=overwrite)
        self.close_segmenter()
        self.extract_surface_mesh(image, overwrite=overwrite)
        self.register_image_to_atlas_NiftyReg(image, overwrite=overwrite)
        self.warp_mesh(image, overwrite=overwrite)
        self.project_thickness_to_atlas(image, overwrite=overwrite)
        # TODO self.project_thickness_to_2D(image, overwrite=overwrite)

    def analyze_multiple_images(self, images, n_workers=10, overwrite=False):
        self.preprocess_parallel(images, n_workers=n_workers, overwrite=overwrite)

        for image in images:
            self.segment_image_and_save_results(image, overwrite=False)
        self.close_segmenter()

        for image in images:
            self.extract_surface_mesh(image, overwrite=overwrite)
            self.register_image_to_atlas_NiftyReg(image, overwrite=overwrite)
            self.warp_mesh(image, overwrite=overwrite)
            self.project_thickness_to_atlas(image, overwrite=overwrite)





