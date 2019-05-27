#!/usr/bin/env python
"""
Created by zhenlinx on 1/31/19
"""
import os
from data.OAI_data import OAIData, OAIImage, OAIPatients
from oai_image_analysis import OAIImageAnalysis
from registration.registers import NiftyReg, AVSMReg
from segmentation.segmenter import Segmenter3DInPatchClassWise

ATLAS_IMAGE_PATH = '/playpen/zyshen/OAI_analysis/atlas/atlas_60_LEFT_baseline_NMI/atlas.nii.gz'
# ATLAS_FC_MESH_PATH = "/playpen/zhenlinx/Code/OAI_analysis/atlas/atlas_60_LEFT_baseline_NMI/atlas_FC_inner_mesh_world.ply"
# ATLAS_TC_MESH_PATH = "/playpen/zhenlinx/Code/OAI_analysis/atlas/atlas_60_LEFT_baseline_NMI/atlas_TC_inner_mesh_world.ply"
# ATLAS_FC_2D_MAP_PATH = "./data/FC_inner_optional_embedded.npy"
# ATLAS_TC_2D_MAP_PATH = "./data/TC_inner_optional_embedded.npy"
ATLAS_FC_MESH_PATH = os.path.join(os.getcwd(),"data/atlas_FC_inner_mesh_world.ply")
ATLAS_TC_MESH_PATH = os.path.join(os.getcwd(),"data/atlas_TC_inner_mesh_world.ply")
ATLAS_FC_2D_MAP_PATH = os.path.join(os.getcwd(), "data/FC_inner_embedded.npy")
ATLAS_TC_2D_MAP_PATH = os.path.join(os.getcwd(), "data/TC_inner_embedded.npy")


def build_default_analyzer(ckpoint_folder=None, use_nifty=True,avsm_path=None, avsm_output_path=None):
    niftyreg_path = "/playpen/zhenlinx/Code/niftyreg/install/bin"
    avsm_path = avsm_path + '/demo'
    register = NiftyReg(niftyreg_path) if use_nifty else AVSMReg(avsm_path,avsm_output_path)
    if not ckpoint_folder:
        ckpoint_folder = "./segmentation/ckpoints/UNet_bias_Nifti_rescaled_LEFT_train1_patch_128_128_32_batch_4_sample_0.01-0.02_BCEWithLogitsLoss_lr_0.001/01272019_212723"
    segmenter_config = dict(
        ckpoint_path=os.path.join(ckpoint_folder, "model_best.pth.tar"),
        training_config_file=os.path.join(ckpoint_folder, "train_config.json"),
        device="cuda",
        batch_size=4,
        overlap_size=(16, 16, 8),
        output_prob=True,
        output_itk=True,
    )
    affine_config = dict(smooth_moving=-1, smooth_ref=-1,
                         max_iterations=10,
                         pv=30, pi=30,
                         num_threads=30)
    bspline_config = dict(
        max_iterations=300,
        # num_levels=3, performed_levels=3,
        smooth_moving=-1, smooth_ref=0,
        sx=4, sy=4, sz=4,
        num_threads=32,
        be=0.1,  # bending energy, second order derivative of deformations (0.01)
    )

    segmenter = Segmenter3DInPatchClassWise(mode="pred", config=segmenter_config)
    analyzer = OAIImageAnalysis(use_nifty)
    analyzer.set_atlas(atlas_image_file=ATLAS_IMAGE_PATH, atlas_FC_mesh_file=ATLAS_FC_MESH_PATH,
                       atlas_TC_mesh_file=ATLAS_TC_MESH_PATH)
    analyzer.set_register(register=register, affine_config=affine_config, bspline_config=bspline_config)
    analyzer.set_segmenter(segmenter=segmenter)
    analyzer.set_preprocess(bias_correct=False, reset_coord=True, normalize_intensity=True, flip_to="LEFT")
    return analyzer


def demo_analyze_single_image(use_nifti,avsm_path=None, avsm_output_path=None,do_clean=False):
    OAI_data_sheet = "./data/SEG_3D_DESS_6visits.csv"
    OAI_data = OAIData(OAI_data_sheet, '/playpen/zhenlinx/data/OAI')
    OAI_data.set_processed_data_paths('/playpen/zyshen/oai_data/OAI_image_analysis',None if use_nifti else 'avsm')
    test_image = OAI_data.get_images(patient_id= [9000099])[0] # 9279291, 9298954,9003380
    analyzer = build_default_analyzer(use_nifty=use_nifti, avsm_path=avsm_path, avsm_output_path=avsm_output_path)
    analyzer.preprocess(test_image, overwrite=False)
    # analyzer.segment_image_and_save_results(test_image, overwrite=False)
    # analyzer.close_segmenter()
    analyzer.extract_surface_mesh(test_image, overwrite=False)
    analyzer.register_image_to_atlas(test_image, True)
    analyzer.warp_mesh(test_image, overwrite=True,do_clean=do_clean)
    #analyzer.project_thickness_to_atlas(test_image, overwrite=False)
    analyzer.set_atlas_2D_map(ATLAS_FC_2D_MAP_PATH,ATLAS_TC_2D_MAP_PATH)
    analyzer.compute_atlas_2D_map(n_jobs=None)
    analyzer.project_thickness_to_2D(test_image, overwrite=False)
    # analyzer.eval_registration_surface_distance(test_image)
    # analyzer.get_surface_distances_eval()


def demo_analyze_cohort(use_nifti,avsm_path=None, avsm_output_path=None,do_clean=False):
    OAI_data_sheet = "data/SEG_3D_DESS_6visits.csv"
    OAI_data = OAIData(OAI_data_sheet, '/playpen-raid/data/OAI')
    OAI_data.set_processed_data_paths('/playpen/zyshen/oai_data/OAI_image_analysis',None if use_nifti else 'avsm')

    patients_ASCII_file_path = "data/Enrollees.txt"
    oai_patients = OAIPatients(patients_ASCII_file_path)
    progression_cohort_patient = oai_patients.filter_patient(V00COHORT='1: Progression')

    progression_cohort_patient_6visits = list(progression_cohort_patient & OAI_data.patient_set)
    progression_cohort_images = OAI_data.get_images(patient_id=progression_cohort_patient_6visits,
                                                    part='LEFT_KNEE')

    subcohort_images = progression_cohort_images[:2]  # 100 patients of progression cohort, 6 visiting each
    analyzer = build_default_analyzer(use_nifty=use_nifti, avsm_path=avsm_path, avsm_output_path=avsm_output_path)

    #analyzer.preprocess_parallel(image_list=subcohort_images, n_workers=32, overwrite=False)
    for test_image in subcohort_images:
        analyzer.segment_image_and_save_results(test_image, overwrite=False)
    analyzer.close_segmenter()

    for i, test_image in enumerate(subcohort_images):
        print("\n[{}] {}\n".format(i, test_image.name))
        analyzer.register_image_to_atlas(test_image, True)
        analyzer.extract_surface_mesh(test_image, overwrite=True)
        analyzer.warp_mesh(test_image, overwrite=True,do_clean=do_clean)
        analyzer.eval_registration_surface_distance(test_image)
        analyzer.set_atlas_2D_map(ATLAS_FC_2D_MAP_PATH, ATLAS_TC_2D_MAP_PATH)
        analyzer.compute_atlas_2D_map(n_jobs=None)
        analyzer.project_thickness_to_atlas(test_image, overwrite=False)
    analyzer.get_surface_distances_eval()


if __name__ == '__main__':
    use_nifti=False
    avsm_path = "/playpen/zyshen/reg_for_analysis"
    avsm_output_path = '/playpen/zyshen/debugs/0414'
    demo_analyze_single_image(use_nifti=use_nifti,avsm_path=avsm_path,avsm_output_path=avsm_output_path,do_clean=True)
    #demo_analyze_cohort(use_nifti=use_nifti,avsm_path=avsm_path,avsm_output_path=avsm_output_path)