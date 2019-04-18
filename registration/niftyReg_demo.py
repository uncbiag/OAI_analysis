#!/usr/bin/env python
"""
Created by zhenlinx on 3/14/19
"""
import os
from registers import NiftyReg


def registration_demo():
    image_dir = '/playpen/zhenlinx/Data/brains/Mindboggle101/histogram_matched/image_in_MNI152_normalized'
    label_dir = '/playpen/zhenlinx/Data/brains/Mindboggle101/histogram_matched/label_31_reID_merged'
    moving_name = 'NKI-RS-22-12'
    target_name = 'NKI-RS-22-18'

    moving_image = os.path.join(image_dir, moving_name +'.nii.gz')
    target_image = os.path.join(image_dir, target_name + '.nii.gz')
    moving_label = os.path.join(label_dir, moving_name + '.nii.gz')
    target_label = os.path.join(label_dir, target_name + '.nii.gz')

    niftyreg_path = "/playpen/zhenlinx/Code/niftyreg/install/bin"
    register = NiftyReg(niftyreg_path)
    bspline_config = dict(
        max_iterations=300,
        # num_levels=3, performed_levels=3,
        smooth_moving=0, smooth_ref=0,
        sx=4, sy=4, sz=4,
        num_threads=32,
        be=0.01,  # bending energy, second order derivative of deformations (0.01)
        # lncc=40
    )

    bspline_output_file = 'bspline_ctp.nii.gz'
    warped_image_path = '{}_to_{}.nii.gz'.format(moving_name, target_name)
    register.register_bspline(target_image, moving_image, warped_image_path=warped_image_path,
                              output_control_point='bspline_ctp.nii.gz',
                              **bspline_config)
    warped_mask = '{}_to_{}_label.nii.gz'.format(moving_name, target_name)
    register.warp_image(target_image, moving_label, bspline_output_file, warped_mask,
                        interp_order=0)

# def eval_dice(mask1, mask2, num_class):
#     import SimpleITK as sitk
#     mask1_np = sitk.GetArrayFromImage(sitk.ReadImage(mask1))
#     mask2_np = sitk.GetArrayFromImage(sitk.ReadImage(mask2))
#     dice_score = []
#     for c in range(1, num_class+1):
#         dice_score.append(compute_dice_coefficient(mask1_np == c, mask2_np == c))

if __name__ == '__main__':
    registration_demo()