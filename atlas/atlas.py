#!/usr/bin/env python
"""
Created by zhenlinx on 10/4/18

Class that build an atlas for OAI data from images with segmentation masks
"""
import sys
import os


sys.path.append(os.path.realpath("../"))
import shutil
import SimpleITK as sitk
import numpy as np


from misc.module_parameters import ParameterDict
from misc.str_ops import replace_extension


def dict_to_ParaDict(dictionary):
    para = ParameterDict()
    para.int = dictionary.copy()
    para.ext = para.int
    return para

def save_dict_to_json(dictionary, json_file):
    para = dict_to_ParaDict(dictionary)
    para.write_JSON(json_file)

def load_jason_to_dict(json_file):
    para = ParameterDict()
    para.load_JSON(json_file)
    return para.ext

def average_masks(average_mask_path, mask_list: list, num_class):
    """
    Average a series of image segmentation masks by majority voting
    :param num_class: number of existing classes in segmentation
    :param average_mask_path: saving path of average mask image
    :param mask_list:list of segmentation mask file paths
    :return:
    """

    for i, mask in enumerate(mask_list):
        seg = sitk.ReadImage(mask)
        seg_np = sitk.GetArrayFromImage(seg)
        if i == 0:
            seg_vote = np.zeros(seg_np.shape + (num_class,))

        for c in range(num_class):
            seg_vote[:, :, :, c] += (seg_np == c)

    avg_mask_np = np.argmax(seg_vote, axis=-1).astype(np.uint8)
    avg_mask = sitk.GetImageFromArray(avg_mask_np)
    avg_mask.CopyInformation(seg)
    sitk.WriteImage(avg_mask, average_mask_path)


class BuildAtlas:
    def __init__(self, register: object, image_list: list, mask_list: list = None, name_list: list = None,
                 atlas_root: str = '', data_root: str = ''):
        """
        :param register: a register object that has necessary registration methods
        :param image_list: a list of image file names
        :param mask_list: a list of segmentation mask file names
        :param atlas_root: where the atlas and intermediate result are saved
        :param data_root: where the input image files are

        """
        self.atlas_root = atlas_root
        self.register = register
        self.image_list = image_list
        self.mask_list = mask_list
        self.name_list = name_list
        self.data_root = data_root
        self.affine_root = None
        self.temp_atlas = None
        self.affine_done = False

    def affine_prealign(self, target_ind, config, folder=None, overwrite=False, num_class=3):
        """
        prealign all image by affine registration

        :param folder: folder where prealigned images are
        :param overwrite: if overwrite is False, registration will not be run when saved results are found
        :param config: configuration of affine registration
        :param target_ind: the index of image in list being the target image for registration
        """
        if folder is None:
            folder = os.path.join(self.atlas_root, "affine_prealign")
        elif not isinstance(folder, str):
            TypeError("the type of folder has to be a string")

        self.affine_root = folder

        if not os.path.isdir(folder):
            os.makedirs(folder)

        config_file = os.path.join(folder, 'config.json')
        if overwrite or (not os.path.isfile(config_file)):
            para = ParameterDict()
            para.int = config.copy()
            para.ext = para.int
            para.write_JSON(config_file)

        # set target image/mask
        target_image = os.path.join(self.data_root, self.image_list[0])
        if self.mask_list:
            target_mask = os.path.join(self.data_root, self.mask_list[0])

        #  copy them into folder
        shutil.copy2(target_image, folder)
        shutil.copy2(target_mask, folder)

        warped_image_list = [os.path.join(folder, self.image_list[0])]
        if self.mask_list:
            warped_mask_list = [os.path.join(folder, self.mask_list[0])]

        # register left image to the target image
        for ind in range(1, len(self.image_list)):
            moving_image = os.path.join(self.data_root, self.image_list[ind])

            warped_image = os.path.join(folder, self.image_list[ind])

            if self.name_list:
                affine_transform_file = os.path.join(folder, self.name_list[ind] + '_affine_transform.txt')
            else:
                affine_transform_file = os.path.join(folder, replace_extension(os.path.basename(moving_image),
                                                                               '_affine_transform.txt'))

            if overwrite or (not os.path.isfile(affine_transform_file)):
                # affine registration
                print("\n\nAffine Register {}th image to 0th image\n\n".format(ind))
                self.register.register_affine(target_image, moving_image, warped_image, affine_transform_file, **config)
            warped_image_list.append(warped_image)

            # self.register.warp_image(target_image, moving_image, affine_transform_file, warped_image, interp_order=3)

            if self.mask_list:
                moving_mask = os.path.join(self.data_root, self.mask_list[ind])
                warped_mask = os.path.join(folder, os.path.basename(self.mask_list[ind]))
                if overwrite or (not os.path.isfile(warped_mask)):
                    # warp mask
                    self.register.warp_image(target_image, moving_mask, affine_transform_file, warped_mask,
                                             interp_order=0)
                warped_mask_list.append(warped_mask)

        self.affine_done = True

        # average prealigned images as the initial atlas
        self.temp_atlas = os.path.join(self.atlas_root, 'atlas_affine.nii.gz')
        if overwrite or (not os.path.isfile(self.temp_atlas)):
            print("Average images after affine registration \n")
            self.register.average_images(self.temp_atlas, images=warped_image_list)

        if self.mask_list:
            self.temp_atlas_mask = os.path.join(self.atlas_root, 'atlas_mask_affine.nii.gz')
            if overwrite and (not os.path.isfile(self.temp_atlas_mask)):
                print("Average segmentations after affine registration \n")
                # if overwrite or (not os.path.isfile(temp_atlas_mask)):
                average_masks(self.temp_atlas_mask, warped_mask_list, num_class=num_class)

    def recursive_deform_reg(self, step, config, overwrite=False, num_class=3):
        # if did not run affine registration and the temp atlas is not set,
        # then use original images instead
        if (not self.affine_done) and (self.temp_atlas is None):
            self.temp_atlas = os.path.join(self.atlas_root, 'atlas_average.nii.gz')
            self.register.average_images(self.temp_atlas, images=[os.path.join(self.data_root, image)
                                                                  for image in self.image_list])
            self.affine_root = self.data_root

        folder = os.path.join(self.atlas_root, "step_{}".format(step))
        target_image = self.temp_atlas

        if not os.path.isdir(folder):
            os.makedirs(folder)

        config_file = os.path.join(folder, 'config.json')
        if overwrite or (not os.path.isfile(config_file)):
            para = ParameterDict()
            para.int = config.copy()
            para.ext = para.int
            para['source_folder'] = self.affine_root
            para.write_JSON(config_file)

        warped_image_list = []
        warped_mask_list = []

        for ind in range(0, len(self.image_list)):
            moving_image = os.path.join(self.affine_root, self.image_list[ind])

            warped_image = os.path.join(folder, self.image_list[ind])

            if self.name_list:
                bspline_transform_file = os.path.join(folder, self.name_list[ind] + '_bspline_transform.nii.gz')
            else:
                bspline_transform_file = os.path.join(folder, replace_extension(os.path.basename(moving_image),
                                                                                '_bspline_transform.nii.gz'))

            if overwrite or (not os.path.isfile(bspline_transform_file)):
                # affine registration
                print("\n\nLoop {} Register {}th image to atlas\n\n".format(step, ind))

                self.register.register_bspline(target_image, moving_image, warped_image,
                                               output_control_point=bspline_transform_file, **config)
            warped_image_list.append(warped_image)

            if self.mask_list:
                moving_mask = os.path.join(self.affine_root, self.mask_list[ind])
                warped_mask = os.path.join(folder, os.path.basename(self.mask_list[ind]))
                if overwrite or (not os.path.isfile(warped_mask)):
                    # warp mask
                    self.register.warp_image(target_image, moving_mask, bspline_transform_file, warped_mask,
                                             interp_order=0)
                warped_mask_list.append(warped_mask)

        self.temp_atlas = os.path.join(self.atlas_root, 'atlas_step_{}.nii.gz'.format(step))
        if overwrite or (not os.path.isfile(self.temp_atlas)):
            print("\nLoop {} Average images after bspline registration \n".format(step))
            self.register.average_images(self.temp_atlas, images=warped_image_list)

        if self.mask_list:
            self.temp_atlas_mask = os.path.join(self.atlas_root, 'atlas_mask_step_{}.nii.gz'.format(step))
            if overwrite or (not os.path.isfile(self.temp_atlas_mask)):
                print("\nLoop {} Average segmentations after bspline registration \n".format(step))
                average_masks(self.temp_atlas_mask, warped_mask_list, num_class=num_class)




if __name__ == '__main__':
    pass
