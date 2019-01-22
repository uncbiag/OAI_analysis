"""
Script for building atlas from OAI segmentation data
"""
import os

from atlas.atlas import BuildAtlas
from registration.registers import NiftyReg
from shape_analysis.cartilage_shape_processing import get_cartilage_surface_mesh_from_segmentation_file
from segmentation.datasets import NiftiDataset


def split_image_by_side(image_list: list):
    """
    split a list of image file by LEFT and RIGHT (in their names)
    :param image_list:
    :return:
    """
    left_list_indices, right_list_indices = [], []
    for i, image in enumerate(image_list):
        if "LEFT" in os.path.basename(image):
            left_list_indices.append(i)
        else:
            right_list_indices.append(i)
    return {"LEFT": left_list_indices, "RIGHT": right_list_indices}

def split_image_by_visit_time(image_list: list, visit_counts: int):
    """
    Seperate images by visiting order
    :param image_list: image list with name in format of patientID_scantime_......
    :param visit_counts: total number of visit for each patient
    :return: a list of lists of indices in original image list for each visit date
    """
    ind_dict = {}
    for i, name in enumerate(image_list):
        patient_id, date = name.split('_')[0:2]
        if patient_id in ind_dict.keys():
            ind_dict[patient_id][date] = i
        else:
            ind_dict[patient_id] = {date: i}

    ind_list = [[] for visit in range(visit_counts)]
    for patient, series in ind_dict.items():
        dates = sorted(series.keys(), key=int)

        for i, date in enumerate(dates):
            ind_list[i].append(series[date])

    # seperate_list = [[image_list[ind] for ind in l] for l in ind_list]

    return ind_list

def OAI_atlas_lncc(affine_config, bspline_config, data_root, image_list, mask_list=None, name_list=None):
    # affine_config = dict(smooth_moving=-3, smooth_ref=-3, pv=50, pi=50, pad=0)
    atlas_root = "/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas"
    register = NiftyReg("/playpen/zhenlinx/Code/niftyreg/install/bin")

    atlas_builder = BuildAtlas(register, image_list[:40], mask_list[:40], name_list=name_list, atlas_root=atlas_root,
                               data_root=data_root)
    atlas_builder.affine_prealign(0, affine_config, overwrite=False)

    for step in range(3, 5):
        atlas_builder.recursive_deform_reg(step=step, config=bspline_config, overwrite=False)


def build_OAI_atlas(register, atlas_root, affine_config, bspline_config, data_root, image_list,
                    mask_list=None, name_list=None, affine_folder=None):

    if not os.path.isdir(atlas_root):
        os.makedirs(atlas_root)

    atlas_builder = BuildAtlas(register, image_list, mask_list, name_list=name_list, atlas_root=atlas_root,
                               data_root=data_root)
    atlas_builder.affine_prealign(0, affine_config, folder=affine_folder, overwrite=False)

    for step in range(1, 11):
        atlas_builder.recursive_deform_reg(step=step, config=bspline_config, overwrite=False)


def build_atlas_experiments():
    """
    build atlas with configs, all images are pre-flipped to the defined orientation (left/right)
    :return:
    """
    num_img = 60
    side_of_knee = 'LEFT'
    visiting_time = 0
    atlas_root = "/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/atlas_{}_{}_baseline_NMI".format(num_img, side_of_knee)
    register = NiftyReg("/playpen/zhenlinx/Code/niftyreg/install/bin")
    image_list_file = os.path.realpath("../data/train1.txt")
    data_root = os.path.realpath("/playpen-raid/zhenlinx/Data/OAI_segmentation/Nifti_rescaled_LEFT")
    affine_config = dict(smooth_moving=-1, smooth_ref=-1,
                         max_iterations=30,
                         pv=50, pi=50,
                         # pad=0,
                         num_threads=32)
    affine_folder = "/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/affine_prealign_{}iter_LEFT".format(
        affine_config['max_iterations'])

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
        # pad=0
    )

    image_list, mask_list, name_list = NiftiDataset.read_image_segmentation_list([image_list_file])

    visit_filter_indices = split_image_by_visit_time(image_list, 2)[visiting_time]
    # side_filter_indices = split_image_by_side(image_list)[side_of_knee]
    # candidate_indices = list(set(visit_filter_indices).intersection(side_filter_indices))
    candidate_indices = visit_filter_indices

    if len(candidate_indices) < num_img -1:
        ValueError("Only {} {} knee images at {}th visit but requested {} images!".format(
            len(candidate_indices), side_of_knee, visiting_time, num_img))

    image_list = [image_list[i] for i in candidate_indices]
    mask_list = [mask_list[i] for i in candidate_indices]
    name_list = [name_list[i] for i in candidate_indices]

    build_OAI_atlas(register, atlas_root, affine_config, bspline_config, data_root, image_list[:num_img],
                    mask_list[:num_img], name_list[:num_img], affine_folder)


def build_atlas_experiments_old():
    num_img = 40
    atlas_root = "/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/atlas_{}baseline_NMI_2".format(num_img)
    register = NiftyReg("/playpen/zhenlinx/Code/niftyreg/install/bin")
    image_list_file = os.path.realpath("../data/train1.txt")
    data_root = os.path.realpath("../data/Nifti_rescaled")
    image_list, mask_list, name_list = NiftiDataset.read_image_segmentation_list([image_list_file])

    baseline_indices = split_image_by_visit_time(image_list, 2)[0]
    image_list = [image_list[i] for i in baseline_indices]
    mask_list = [mask_list[i] for i in baseline_indices]
    name_list = [name_list[i] for i in baseline_indices]

    affine_config = dict(smooth_moving=-1, smooth_ref=-1,
                         max_iterations=30,
                         pv=50, pi=50,
                         # pad=0,
                         num_threads=32)
    affine_folder = "/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/affine_prealign_{}iter".format(
        affine_config['max_iterations'])

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
        # pad=0
    )

    build_OAI_atlas(register, atlas_root, affine_config, bspline_config, data_root, image_list[:num_img],
                    mask_list[:num_img], name_list[:num_img], affine_folder)


def build_OAI_ZIB_atlas():
    """
        build atlas with configs, all images are pre-flipped to the defined orientation (left/right)
        :return:
        """
    num_img = 40
    side_of_knee = 'LEFT'
    visiting_time = 0
    atlas_root = "/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/atlas_{}_{}_baseline_NMI".format(num_img,
                                                                                                      side_of_knee)
    register = NiftyReg("/playpen/zhenlinx/Code/niftyreg/install/bin")
    image_list_file = os.path.realpath("../data/train1.txt")
    data_root = os.path.realpath("/playpen-raid/zhenlinx/Data/OAI_segmentation/Nifti_rescaled_LEFT")

    affine_config = dict(smooth_moving=-1, smooth_ref=-1,
                         max_iterations=30,
                         pv=50, pi=50,
                         # pad=0,
                         num_threads=32)
    affine_folder = "/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/affine_prealign_{}iter_LEFT".format(
        affine_config['max_iterations'])

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
        # pad=0
    )

    image_list, mask_list, name_list = NiftiDataset.read_image_segmentation_list([image_list_file])

    visit_filter_indices = split_image_by_visit_time(image_list, 2)[visiting_time]
    # side_filter_indices = split_image_by_side(image_list)[side_of_knee]
    # candidate_indices = list(set(visit_filter_indices).intersection(side_filter_indices))
    candidate_indices = visit_filter_indices

    if len(candidate_indices) < num_img - 1:
        ValueError("Only {} {} knee images at {}th visit but requested {} images!".format(
            len(candidate_indices), side_of_knee, visiting_time, num_img))

    image_list = [image_list[i] for i in candidate_indices]
    mask_list = [mask_list[i] for i in candidate_indices]
    name_list = [name_list[i] for i in candidate_indices]

    build_OAI_atlas(register, atlas_root, affine_config, bspline_config, data_root, image_list[:num_img],
                    mask_list[:num_img], name_list[:num_img], affine_folder)

def generate_mesh_for_atlas():
    atlas_folder = '/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/atlas_60_LEFT_baseline_NMI'
    atlas_mask= 'atlas_mask_step_10.nii.gz'

    FC_mesh_main, TC_mesh_main = get_cartilage_surface_mesh_from_segmentation_file(os.path.join(atlas_folder, atlas_mask),
                                                                                   thickness=False,
                                                                                   save_path_FC='test_analysis/atlas_FC_mesh_world.ply',
                                                                                   save_path_TC='test_analysis/atlas_TC_mesh_world.ply',
                                                                                   prob=False, coord='nifti')

    # segmentation = sitk_read_image(atlas_mask, atlas_folder)
    pass


def warp_mesh_to_atlas():
    atlas_image =  '/playpen-raid/zhenlinx/Data/OAI_segmentation/atlas/atlas_60_LEFT_baseline_NMI/atlas_mask_step_10.nii.gz'

    # atlas_mask = ''
    atlas_mesh = './test_analysis/atlas_FC_mesh_world.ply'
    moving_image = './test_analysis/9007827_20041006_SAG_3D_DESS_LEFT_016610263603_image.nii.gz'
    moving_mask = './test_analysis/9007827_20041006_SAG_3D_DESS_LEFT_016610263603_label_all.nii.gz'
    moving_mesh_FC = './test_analysis/9007827_20041006_SAG_3D_DESS_LEFT_016610263603_mesh_FC.ply'
    moving_mesh_TC = './test_analysis/9007827_20041006_SAG_3D_DESS_LEFT_016610263603_mesh_TC.ply'
    get_cartilage_surface_mesh_from_segmentation_file(moving_mask, thickness=True, prob=False, coord='nifti',
                                                      save_path_FC=moving_mesh_FC, save_path_TC=moving_mesh_TC)

    affine_transform = './test_analysis/9007827_20041006_SAG_3D_DESS_LEFT_016610263603_affine_transform.txt'
    non_rigid_transform = './test_analysis/9007827_20041006_SAG_3D_DESS_LEFT_016610263603_bspline_transform.nii.gz'

    register = NiftyReg("/playpen/zhenlinx/Code/niftyreg/install/bin")

    inv_affine_transform = './test_analysis/9007827_20041006_SAG_3D_DESS_LEFT_016610263603_affine_transform_inverted.txt'
    register.invert_affine(affine_transform, inv_affine_transform)
    register.warp_mesh(moving_mesh_FC, inv_affine_transform, moving_image, inWorld=True)
    register.warp_mesh(moving_mesh_TC, inv_affine_transform, moving_image, inWorld=True)

    inv_nonrigid_transform = './test_analysis/9007827_20041006_SAG_3D_DESS_LEFT_016610263603_bspline_transform_inverted.nii.gz'
    register.invert_nonrigid(non_rigid_transform=non_rigid_transform, reference_image=atlas_image, moving_image = moving_image, inverted_transform=inv_nonrigid_transform,)
    warped_mesh_FC = register.warp_mesh(moving_mesh_FC, inv_nonrigid_transform, moving_image, inWorld=True)
    warped_mesh_TC = register.warp_mesh(moving_mesh_TC, inv_nonrigid_transform, moving_image, inWorld=True)


if __name__ == '__main__':
    # import pymesh
    # build_OAI_ZIB_atlas()
    # generate_mesh_for_atlas()
    # warp_mesh_to_atlas()

    test_map_thickness()

    # FC_mesh_main = pymesh.load_mesh('./test_analysis/9007827_20041006_SAG_3D_DESS_LEFT_016610263603_mesh_FC.ply')
    # FC_mesh_main_affine = pymesh.load_mesh('./test_analysis/9007827_20041006_SAG_3D_DESS_LEFT_016610263603_mesh_FC_warped_with_9007827_20041006_SAG_3D_DESS_LEFT_016610263603_affine_transform_inverted.ply')
    # FC_mesh_main_nr = pymesh.load_mesh('./test_analysis/9007827_20041006_SAG_3D_DESS_LEFT_016610263603_mesh_FC_warped_with_9007827_20041006_SAG_3D_DESS_LEFT_016610263603_bspline_transform_inverted.ply')
    # TC_mesh_main = pymesh.load_mesh('./test_analysis/9007827_20041006_SAG_3D_DESS_LEFT_016610263603_mesh_TC.ply')
    # TC_mesh_main_affine = pymesh.load_mesh('./test_analysis/9007827_20041006_SAG_3D_DESS_LEFT_016610263603_mesh_TC_warped_with_9007827_20041006_SAG_3D_DESS_LEFT_016610263603_affine_transform_inverted.ply')
    # TC_mesh_main_nr = pymesh.load_mesh('./test_analysis/9007827_20041006_SAG_3D_DESS_LEFT_016610263603_mesh_TC_warped_with_9007827_20041006_SAG_3D_DESS_LEFT_016610263603_bspline_transform_inverted.ply')
    #
    # app = vv.use()
    # a1 = vv.subplot(311)
    # FC_vis = vv.mesh((FC_mesh_main.vertices), FC_mesh_main.faces, values=FC_mesh_main.get_attribute('vertex_thickness'))
    # TC_vis = vv.mesh((TC_mesh_main.vertices), TC_mesh_main.faces, values=TC_mesh_main.get_attribute('vertex_thickness'))
    # FC_vis.colormap = vv.CM_JET
    # TC_vis.colormap = vv.CM_JET
    # vv.colorbar()
    # a2 = vv.subplot(312)
    # FC_vis = vv.mesh((FC_mesh_main_affine.vertices), FC_mesh_main_affine.faces, values=FC_mesh_main_affine.get_attribute('vertex_thickness'))
    # TC_vis = vv.mesh((TC_mesh_main_affine.vertices), TC_mesh_main_affine.faces, values=TC_mesh_main_affine.get_attribute('vertex_thickness'))
    # FC_vis.colormap = vv.CM_JET
    # TC_vis.colormap = vv.CM_JET
    # vv.colorbar()
    # a3 = vv.subplot(313)
    # FC_vis = vv.mesh((FC_mesh_main_nr.vertices), FC_mesh_main_nr.faces, values=FC_mesh_main_nr.get_attribute('vertex_thickness'))
    # TC_vis = vv.mesh((TC_mesh_main_nr.vertices), TC_mesh_main_nr.faces, values=TC_mesh_main_nr.get_attribute('vertex_thickness'))
    # FC_vis.colormap = vv.CM_JET
    # TC_vis.colormap = vv.CM_JET
    # vv.colorbar()
    # app.Run()