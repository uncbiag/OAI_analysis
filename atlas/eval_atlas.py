#!/usr/bin/env python
"""
Script to evaluate atlas by:
1. measure the overlapping/distance metric on segmentations of any pair of registered images
2. measure the overlapping/distance metric on segmentations of one register images and the atlas

Evaluations are run on both images used for constructing the atlas and unseen images

Created by zhenlinx on 10/27/18
"""
import os
import sys
import SimpleITK as sitk
from build_atlas import *
sys.path.append(os.path.realpath("../"))
sys.path.append(os.path.realpath("../lib"))
from surface_distance.metrics import *
from glob import glob
from registration.registers import NiftyReg
from segmentation.datasets import NiftiDataset
from misc.module_parameters import  save_dict_to_json

class EvaluateAtlas():
    def __init__(self, atlas_image, atlas_mask_file, number_class=1):
        self.atlas_image_path = atlas_image
        self.atlas_mask_path = atlas_mask_file
        self.num_foreground_class = number_class

    @staticmethod
    def evaluate_segmentations_between_image_and_atlas(atlas_mask_path, segmentation_path_list, num_foreground_class):
        """
        Measure the metric between segmentation of atlas and that of image registered to atlas
        Assume that segmentation images store array of logits which represent class indices in which 0 means background
        :param num_foreground_class: number of foreground class in segmentation image.
        :param segmentation_path_list: path of segmentations of register images
        :return:
        """
        metric_per_class_per_pair = {i:{'dice':[], 'avg_surf_dist':[], "hausdorff_100":[], "hausdorff_95":[]} for i in range(
            num_foreground_class)}
        metric_per_class_avg = {i: {'dice': [], 'avg_surf_dist': [], "hausdorff_100": [], "hausdorff_95": []} for i in range(
            num_foreground_class)}

        atlas_seg = sitk.ReadImage(atlas_mask_path)
        atlas_seg_np = sitk.GetArrayFromImage(atlas_seg)
        atlas_binary_masks = [atlas_seg_np == i+1 for i in range(num_foreground_class)]
        spacing = atlas_seg.GetSpacing()[::-1]

        # compute metric between atlas and image
        for segmentation_path in segmentation_path_list:
            segmentation_np = sitk.GetArrayFromImage(sitk.ReadImage(segmentation_path))
            image_binary_masks = [segmentation_np == i + 1 for i in range(num_foreground_class)]
            for c in range(num_foreground_class):
                metric_per_class_per_pair[c]["dice"].append(compute_dice_coefficient(atlas_binary_masks[c], image_binary_masks[c]))

                surface_distances = compute_surface_distances(atlas_binary_masks[c], image_binary_masks[c], spacing)
                metric_per_class_per_pair[c]["avg_surf_dist"].append(compute_average_surface_distance(surface_distances))
                metric_per_class_per_pair[c]["hausdorff_100"].append(compute_robust_hausdorff(surface_distances, 100))
                metric_per_class_per_pair[c]["hausdorff_95"].append(compute_robust_hausdorff(surface_distances, 95))

        # compute average and std
        for c in range(num_foreground_class):
            for key in metric_per_class_per_pair[c]:
                if key == 'avg_surf_dist':
                    metric_per_class_avg[c][key] = {"avg": (np.mean([dist[0] for dist in metric_per_class_per_pair[c][key]]),
                                                    np.mean([dist[1] for dist in metric_per_class_per_pair[c][key]])),
                                                    "std": (
                                                    np.std([dist[0] for dist in metric_per_class_per_pair[c][key]]),
                                                    np.std([dist[1] for dist in metric_per_class_per_pair[c][key]]))
                                                    }
                else:
                    metric_per_class_avg[c][key] = {"avg": np.mean(metric_per_class_per_pair[c][key]),
                                                    "std": np.std(metric_per_class_per_pair[c][key])}

        return metric_per_class_per_pair, metric_per_class_avg

    @staticmethod
    def evaluate_segmentations_between_images(segmentation_path_list, num_foreground_class):
        """
        measure the metric between all segmentations pairs of images registered to atlas
        :param num_foreground_class: number of foreground class in segmentation image.
        :param segmentation_path_list: path of segmentations of register images
        :return:
        """
        metric_per_class_per_pair = {i: {'dice': [], 'avg_surf_dist': [], "hausdorff_100": [], "hausdorff_95": []}
                                     for i in range(num_foreground_class)}
        metric_per_class_avg = {i: {'dice': [], 'avg_surf_dist': [], "hausdorff_100": [], "hausdorff_95": []}
                                for i in range(num_foreground_class)}
        num_seg = len(segmentation_path_list)
        binary_masks = []
        spacing = None

        # read images
        for i in range(num_seg):
            seg = sitk.ReadImage(segmentation_path_list[i])
            seg_np = sitk.GetArrayFromImage(seg)
            binary_masks.append([seg_np == i + 1 for i in range(num_foreground_class)])
            if i == 0:
                spacing = seg.GetSpacing()[::-1]

        # compute metric cross all possible images
        for i in range(num_seg):
            for j in range(i+1, num_seg):
                for c in range(num_foreground_class):
                    metric_per_class_per_pair[c]["dice"].append(
                        compute_dice_coefficient(binary_masks[i][c], binary_masks[j][c]))

                    surface_distances = compute_surface_distances(binary_masks[i][c], binary_masks[j][c], spacing)
                    metric_per_class_per_pair[c]["avg_surf_dist"].append(
                        compute_average_surface_distance(surface_distances))
                    metric_per_class_per_pair[c]["hausdorff_100"].append(compute_robust_hausdorff(surface_distances, 100))
                    metric_per_class_per_pair[c]["hausdorff_95"].append(compute_robust_hausdorff(surface_distances, 95))

        # compute average and std
        for c in range(num_foreground_class):
            for key in metric_per_class_per_pair[c]:
                if key == 'avg_surf_dist':
                    metric_per_class_avg[c][key] = {"avg": np.mean([dist[0] for dist in metric_per_class_per_pair[c][key]]
                                                           + [dist[1] for dist in metric_per_class_per_pair[c][key]]),
                                                    "std": np.std([dist[0] for dist in metric_per_class_per_pair[c][key]]
                                                                   + [dist[1] for dist in metric_per_class_per_pair[c][key]])
                                                    }

                else:
                    metric_per_class_avg[c][key] = {"avg": np.mean(metric_per_class_per_pair[c][key]),
                                                    "std": np.std(metric_per_class_per_pair[c][key])}

        return metric_per_class_per_pair, metric_per_class_avg

    def eval_segmentation_metrics(self, registered_seg_list, overwrite=False):

        result_dir = os.path.dirname(registered_seg_list[0])

        # evaluate between atlas and images
        atlas_image_metrics_file = os.path.join(result_dir,"metric_to_atlas.json")
        atlas_image_metrics_avg_file = os.path.join(result_dir,"metric_to_atlas_avg.json")
        cross_image_metrics_file = os.path.join(result_dir,"metric_cross_images.json")
        cross_image_metrics_avg_file = os.path.join(result_dir,"metric_cross_images_avg.json")


        print("Computing atlas-image segmentation metrics")
        if not (os.path.isfile(atlas_image_metrics_file) and os.path.isfile(atlas_image_metrics_avg_file)) or overwrite:
            atlas_image_metrics, atlas_image_metrics_avg = self.evaluate_segmentations_between_image_and_atlas(
                self.atlas_mask_path, registered_seg_list, self.num_foreground_class)
            save_dict_to_json(atlas_image_metrics, atlas_image_metrics_file)
            save_dict_to_json(atlas_image_metrics_avg, atlas_image_metrics_avg_file)

        print("Computing image-image segmentation metrics")
        # evaluate cross image pairs
        if not (os.path.isfile(cross_image_metrics_file) and os.path.isfile(
                cross_image_metrics_avg_file)) or overwrite:
            cross_image_metrics, cross_image_metrics_avg = self.evaluate_segmentations_between_images(
                registered_seg_list, self.num_foreground_class)
            save_dict_to_json(cross_image_metrics, cross_image_metrics_file)
            save_dict_to_json(cross_image_metrics_avg, cross_image_metrics_avg_file)


    def register_to_atlas(self, register, affine_config, bspline_config, moving_image_list, moving_seg_list=None,
                          moving_name_list=None, result_root=None, image_root=None, overwrite=False):
        mask_registered_to_atlas_list = []

        if not result_root:
            result_root = os.path.join(os.path.dirname(self.atlas_image_path), 'test')
        affine_folder = os.path.join(result_root, "affine")
        bspline_folder = os.path.join(result_root, "bspline")

        if not os.path.isdir(affine_folder):
            os.makedirs(affine_folder)
        if not os.path.isdir(bspline_folder):
            os.makedirs(bspline_folder)

        affine_config_file = os.path.join(affine_folder, 'affine_config.json')
        if overwrite or (not os.path.isfile(affine_config_file)):
            para = ParameterDict()
            para.int = affine_config.copy()
            para.ext = para.int
            para.write_JSON(affine_config_file)

        bspline_config_file = os.path.join(bspline_folder, 'bspline_config.json')
        if overwrite or (not os.path.isfile(bspline_config_file)):
            para = ParameterDict()
            para.int = bspline_config.copy()
            para.ext = para.int
            para.write_JSON(bspline_config_file)

        for i, moving_image in enumerate(moving_image_list):
            moving_seg = moving_seg_list[i]
            moving_name = moving_name_list[i]
            moving_image_path = os.path.join(image_root, moving_image)
            moving_seg_path = os.path.join(image_root, moving_seg)

            affine_warped_image_path = os.path.join(affine_folder, moving_image)
            affine_warped_seg_path = os.path.join(affine_folder, moving_seg)
            affine_transform_file = os.path.join(affine_folder, moving_name + '_affine_transform.txt')

            print("\n\nProcess {}th image\n\n".format(i))

            if overwrite or (not os.path.isfile(affine_transform_file)):
                print("\n\nAffine Register {}th image to atlas\n\n".format(i))
                register.register_affine(self.atlas_image_path, moving_image_path,
                                         out_affine_file=affine_transform_file,
                                         warped_image_path=affine_warped_image_path,
                                          **affine_config)
            if overwrite or (not os.path.isfile(affine_warped_seg_path)):
                # warp mask
                print("\n\nAffine warp segmentation of {}th image to atlas space\n\n".format(i))

                register.warp_image(self.atlas_image_path, moving_seg_path, affine_transform_file, affine_warped_seg_path,
                                    interp_order=0)

            bspline_warped_image_path = os.path.join(bspline_folder, moving_image)
            bspline_warped_seg_path = os.path.join(bspline_folder, moving_seg)
            bspline_transform_file = os.path.join(bspline_folder, moving_name + '_bspline_transform.nii.gz')
            bspline_deformfield_file = os.path.join(bspline_folder, moving_name + '_bspline_deformfield.nii.gz')
            if overwrite or (not os.path.isfile(bspline_transform_file)):
                print("\n\nBSpline Register {}th image to atlas\n\n".format(i))
                register.register_bspline(self.atlas_image_path, moving_image_path, bspline_warped_image_path,
                                          init_affine_file=affine_transform_file,
                                          output_control_point=bspline_transform_file,
                                          **bspline_config)

            if overwrite or (not os.path.isfile(bspline_warped_seg_path)):
                # warp mask
                print("\n\nBSpline warp segmentation of {}th image to atlas space\n\n".format(i))
                register.warp_image(self.atlas_image_path, moving_seg_path, bspline_transform_file, bspline_warped_seg_path,
                                    interp_order=0)

            mask_registered_to_atlas_list.append(bspline_warped_seg_path)

            # if overwrite or (not os.path.isfile(bspline_deformfield_file)):
            #     print("\n\nGenerate deformation field from BSpline control points file\n\n")
            #     register.transform_to_deformation_field(bspline_transform_file, bspline_deformfield_file, self.atlas_image)

        return mask_registered_to_atlas_list


def eval_atlas_on_test_image(atlas_dir, image_root, overwrite=False):
    register = NiftyReg("/playpen/zhenlinx/Code/niftyreg/install/bin")
    image_list_file = os.path.realpath("../data/test1.txt")
    image_list, mask_list, name_list = NiftiDataset.read_image_segmentation_list([image_list_file])

    affine_config = dict(smooth_moving=-1, smooth_ref=0,
                         max_iterations=10,
                         pv=50, pi=50,
                         pad=0,
                         num_threads=32)

    bspline_config = dict(
        max_iterations=300,
        # num_levels=3, performed_levels=3,
        smooth_moving=-1, smooth_ref=0,
        sx=4, sy=4, sz=4,
        # platf=0, gpu_id=1,
        num_threads=32,
        # lncc=40,
        # bending energy, second order derivative of deformations (0.01)
        be=0.1,
        pad=0
    )

    atlas_image = os.path.join(atlas_dir, "atlas_step_10.nii.gz")
    atlas_mask = os.path.join(atlas_dir, "atlas_mask_step_10.nii.gz")

    eval_atlas = EvaluateAtlas(atlas_image, atlas_mask, number_class=2)

    print("Register Test images in \n{}\n to atlas {}\n".format(image_root, atlas_image))
    mask_registered_list = eval_atlas.register_to_atlas(register, affine_config, bspline_config, image_list, moving_seg_list=mask_list,
                      moving_name_list=name_list,
                      image_root=image_root, overwrite=overwrite)

    eval_atlas.eval_segmentation_metrics(mask_registered_list, overwrite=overwrite)



def run_register_unseen_images_to_atlas():

    atlas_paths = ["/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/atlas_40baseline_NMI_2",
                   "/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/atlas_40baseline_NMI_3",
                   "/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/atlas_30_LEFT_baseline_NMI",
                   "/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/atlas_40_LEFT_baseline_NMI",
                   "/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/atlas_60_LEFT_baseline_NMI"]


    for atlas_folder in atlas_paths:
        if "LEFT" in atlas_folder:
            image_root = "/playpen-raid/zhenlinx/Data/OAI_segmentation/Nifti_rescaled_LEFT"
        elif "RIGHT" in atlas_folder:
            image_root = "/playpen-raid/zhenlinx/Data/OAI_segmentation/Nifti_rescaled_RIGHT"
        else:
            image_root = "/playpen-raid/zhenlinx/Data/OAI_segmentation/Nifti_rescaled"

        eval_atlas_on_test_image(atlas_folder, image_root, overwrite=False)


def run_evaluate_segmentations(overwrite=False):
    # image_list_file = os.path.realpath("../data/test1.txt")
    # register_image_folder = "/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/atlas_40baseline_NMI_2/test/bspline"
    # image_list, mask_list, name_list = NiftiDataset.read_image_segmentation_list([image_list_file],
    #                                                                              register_image_folder)


    # seg_dir = "/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/atlas_40baseline_NMI_2/test/bspline"
    # atlas_seg_path = "/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/atlas_40baseline_NMI_2/atlas_mask_step_10.nii.gz"
    atlas_paths = [
                # "/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/atlas_40baseline_NMI_2",
                   "/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/atlas_40baseline_NMI_3",
                   # "/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/atlas_30_LEFT_baseline_NMI",
                   # "/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/atlas_40_LEFT_baseline_NMI",
                   # "/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/atlas_60_LEFT_baseline_NMI"
                   ]
    seg_dirs = ["step_10", "test/bspline"]
    # seg_dirs = ["test/bspline"]

    for path in atlas_paths:
        for seg_dir in seg_dirs:
            seg_dir = os.path.join(path, seg_dir)
            atlas_seg_path = os.path.join(path, "atlas_mask_step_10.nii.gz")
            mask_list = glob(os.path.join(seg_dir, "*_label_all.nii.gz"))

            # evaluate between atlas and images
            atlas_image_metrics_file = "metric_to_atlas.json"
            atlas_image_metrics_avg_file = "metric_to_atlas_avg.json"
            cross_image_metrics_file = "metric_cross_images.json"
            cross_image_metrics_avg_file = "metric_cross_images_avg.json"

            if not (os.path.isfile(atlas_image_metrics_file) and os.path.isfile(atlas_image_metrics_avg_file)) or overwrite:
                atlas_image_metrics, atlas_image_metrics_avg = EvaluateAtlas.evaluate_segmentations_between_image_and_atlas(
                    mask_list, atlas_seg_path, num_foreground_class=2)
                save_dict_to_json(atlas_image_metrics, os.path.join(seg_dir, atlas_image_metrics_file))
                save_dict_to_json(atlas_image_metrics_avg, os.path.join(seg_dir, atlas_image_metrics_avg_file))

            # evaluate cross image pairs
            if not (os.path.isfile(cross_image_metrics_file) and os.path.isfile(
                    cross_image_metrics_avg_file)) or overwrite:
                cross_image_metrics, cross_image_metrics_avg = EvaluateAtlas.evaluate_segmentations_between_images(
                    mask_list, num_foreground_class=2)
                save_dict_to_json(cross_image_metrics, os.path.join(seg_dir, cross_image_metrics_file))
                save_dict_to_json(cross_image_metrics_avg, os.path.join(seg_dir, cross_image_metrics_avg_file))





if __name__ == '__main__':
    # run_evaluate_segmentations()
    # run_register_unseen_images_to_atlas()
    pass