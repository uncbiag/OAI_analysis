#!/usr/bin/env python
"""
Created by zhenlinx on 1/27/19
"""
import os
import sys

sys.path.append(os.path.realpath(".."))
from pipelines import build_default_analyzer
from data.OAI_data import OAIImage, OAIData

import unittest

class test_pipeline(unittest.TestCase):

    def test_single_image_pipeline(self):
        ckpoint_folder = "../segmentation/ckpoints/UNet_bias_Nifti_rescaled_LEFT_train1_patch_128_128_32_batch_4_sample_0.01-0.02_BCEWithLogitsLoss_lr_0.001/01272019_212723"
        analyzer = build_default_analyzer(ckpoint_folder)
        test_image = OAIImage()
        test_image.folder = "./test_data"
        # test_image.preprocessed_image_file = "/playpen-raid/zhenlinx/Data/OAI/9010952/MR_SAG_3D_DESS/LEFT KNEE/12 MONTH/image_normalized.nii.gz"
        test_image.preprocessed_image_file = "./test_data/test_image.nii.gz"

        analyzer.segment_image_and_save_results(test_image, overwrite=True)
        analyzer.close_segmenter()
        analyzer.extract_surface_mesh(test_image, overwrite=False)

        analyzer.register_image_to_atlas_NiftyReg(test_image, True)
        analyzer.project_thickness_to_atlas(test_image, overwrite=False)
