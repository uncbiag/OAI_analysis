#!/usr/bin/env python
"""
Created by zhenlinxu on 01/04/2019
"""
import os
import sys
sys.path.append(os.path.realpath(".."))

import SimpleITK as sitk


class OAIImageAnalysis:
    def __init__(self):
        self.segmenter = None
        self.register = None
        self.atlas_image = None
        self.atlas_mask = None
        self.atlas_FC_mesh = None
        self.atlas_TC_mesh = None

        pass

    def run_segmentation(self, image, save_folder):
        self.segmentor(image)

    @staticmethod
    def _segment_image_and_save_results(image, segmentor, overwrite=False):
        """
        Segment image and save the integer mask and probmap.
        The results are saved into a folder named by the segmentor's name under the path of image file.
        :param image(OAIImage object)
        :param overwrite: if overwrite existing segmentations
        :return: None
        """
        # generate segmentation folder and files path
        image.segmentation_dir = os.path.join(image.folder, segmentor.name)
        mask_path = os.path.join(image.segmentation_dir, 'cartilage_FC_TC_mask.nii.gz')
        prob_map_path = os.path.join(image.segmentation_dir, 'cartilage_FC_TC_probmap.nii.gz')

        if (not overwrite) and os.path.isfile(mask_path) and os.path.isfile(prob_map_path):
            print("Segmentations found at {}".format(image.folder))
        else:
            print("Segmenting {}".format(image.folder))
            image = sitk.ReadImage(image.normalized_image_path)
            segmentation = segmentor.seg_image(image, if_output_prob_map=True)
            if not os.path.exists(image.segmentation_dir):
                os.makedirs(image.segmentation_dir)
            sitk.WriteImage(segmentation[0], mask_path)

            prop_map_image = sitk.Compose([segmentation[1], segmentation[2], segmentation[3]])
            sitk.WriteImage(prop_map_image, prob_map_path)

        image.cartilage_segmentation_path = mask_path
        image.cartilage_probmap_path = prob_map_path


    def run_registration(self, image, save_folder):
        self.register(image)

    def extract_surface_mesh(self):
        pass

    def warp_surface_mesh(self):
        pass

if __name__ == '__main__':
    main()
