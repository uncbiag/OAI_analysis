#!/usr/bin/env python
"""
Created by zhenlinx on 1/25/19
"""
import os
import segmentation.datasets as data3d
from segmentation.segmenter import Segmenter3DInPatchClassWise
def pred():

    pred_config = dict(
        ckpoint_path = "./ckpoints/UNet_bias_Nifti_rescaled_toLeft_train1_patch_128_128_32_batch_4_sample_0.01-0.02_BCEWithLogitsLoss_lr_0.001/01242019_232540/"
                       "checkpoint.pth.tar",
        training_config_file = "./ckpoints/UNet_bias_Nifti_rescaled_toLeft_train1_patch_128_128_32_batch_4_sample_0.01-0.02_BCEWithLogitsLoss_lr_0.001/01242019_232540/"
                          "train_config.json",
        device = "cuda",
        batch_size = 4,
        overlap_size =(16, 16, 8),
        output_prob=True,
        output_itk=True,
    )

    data_root = "/playpen/zhenlinx/Data/OAI_segmentation"
    validation_list_file = os.path.join(data_root, "validation1.txt")
    valid_data_dir = os.path.join(data_root, "Nifti_rescaled")
    validation_data = data3d.NiftiDataset(validation_list_file, valid_data_dir, mode="pred",
                                          preload=False)

    segmenter = Segmenter3DInPatchClassWise(mode="pred", config=pred_config)
    for image, _, name in validation_data:
        pred_FC, pred_TC = segmenter.segment(image)

if __name__ == '__main__':
    pred()