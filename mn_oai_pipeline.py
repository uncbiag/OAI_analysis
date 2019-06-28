#!/usr/bin/env python
"""
Created by zhenlinx on 1/31/19
Adapted by mn on 06/28/19
"""
import os
from data.OAI_data import OAIData, OAIImage, OAIPatients
from oai_image_analysis import OAIImageAnalysis
from registration.registers import NiftyReg, AVSMReg
from segmentation.segmenter import Segmenter3DInPatchClassWise
import random
import torch
import numpy as np
import sys

import module_parameters as pars

# global parameters
PARAMS = pars.ParameterDict()

# some settings for which we do not need user input
PARAMS['python_executable'] = (sys.executable,'used python interpreter')
PARAMS['atlas_fc_mesh_path'] = (os.path.join(os.getcwd(),'data/atlas_FC_inner_mesh_world.ply'), 'Atlas inner mesh for femoral cartilage.')
PARAMS['atlas_tc_mesh_path'] = (os.path.join(os.getcwd(),'data/atlas_TC_inner_mesh_world.ply'), 'Atlas inner mesh for tibial cartilage.')
PARAMS['atlas_fc_2d_map_path'] = (os.path.join(os.getcwd(), 'data/FC_inner_embedded.npy'), 'Computed embedding for the inner atlas femoral cartilage mesh.')
PARAMS['atlas_tc_2d_map_path'] = (os.path.join(os.getcwd(), 'data/TC_inner_embedded.npy'), 'Computed embedding for the inner atlas tibial cartilage mesh.')
PARAMS['oai_data_sheet'] = ('data/SEG_3D_DESS_6visits.csv','The data sheet describing all the images.')
PARAMS['oai_enrollees'] = ('data/Enrollees.txt','Patient ID file for the OAI data.')


def build_default_analyzer(ckpoint_folder=None, use_nifty=True,avsm_path=None):
    niftyreg_path = PARAMS['nifty_reg_directory']
    avsm_path = avsm_path + '/demo'
    register = NiftyReg(niftyreg_path) if use_nifty else AVSMReg(avsm_path=avsm_path,python_executable=PARAMS['python_executable'])
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
    analyzer.set_atlas(atlas_image_file=PARAMS['atlas_image'], atlas_FC_mesh_file=PARAMS['atlas_fc_mesh_path'],
                       atlas_TC_mesh_file=PARAMS['atlas_tc_mesh_path'])
    analyzer.set_register(register=register, affine_config=affine_config, bspline_config=bspline_config)
    analyzer.set_segmenter(segmenter=segmenter)
    analyzer.set_preprocess(bias_correct=False, reset_coord=True, normalize_intensity=True, flip_to="LEFT")
    return analyzer


def demo_analyze_single_image(use_nifti,avsm_path=None,do_clean=False):
    OAI_data_sheet = PARAMS['oai_data_sheet']
    OAI_data = OAIData(OAI_data_sheet, PARAMS['oai_data_directory'])
    OAI_data.set_processed_data_paths( PARAMS['output_directory'],None if use_nifti else 'avsm')
    test_image = OAI_data.get_images(patient_id= [9279291])[0] # 9279291, 9298954,9003380
    analyzer = build_default_analyzer(use_nifty=use_nifti, avsm_path=avsm_path)
    analyzer.preprocess(test_image, overwrite=False)
    # analyzer.segment_image_and_save_results(test_image, overwrite=False)
    # analyzer.close_segmenter()
    analyzer.extract_surface_mesh(test_image, overwrite=False)
    analyzer.register_image_to_atlas(test_image, True)
    analyzer.warp_mesh(test_image, overwrite=True,do_clean=do_clean)
    #analyzer.project_thickness_to_atlas(test_image, overwrite=False)
    analyzer.set_atlas_2D_map(PARAMS['atlas_fc_2d_map_path'], PARAMS['atlas_tc_2d_map_path'])
    analyzer.compute_atlas_2D_map(n_jobs=None)
    analyzer.project_thickness_to_2D(test_image, overwrite=False)
    # analyzer.eval_registration_surface_distance(test_image)
    # analyzer.get_surface_distances_eval()


def analyze_cohort(use_nifti,avsm_path=None, do_clean=False, overwrite=False,task_id=None,data_division_interval=None,data_division_offset=None):

    OAI_data_sheet = PARAMS['oai_data_sheet']
    OAI_data = OAIData(OAI_data_sheet, PARAMS['oai_data_directory'])
    # we do not create the directories here, as we want to do this on the fly
    task_name = None if use_nifti else 'avsm'
    OAI_data.set_processed_data_paths_without_creating_image_directories( PARAMS['output_directory'],task_name=task_name)

    patients_ASCII_file_path = PARAMS['oai_enrollees']
    oai_patients = OAIPatients(patients_ASCII_file_path)
    progression_cohort_patient = oai_patients.filter_patient(V00COHORT='1: Progression')

    progression_cohort_patient_6visits = list(progression_cohort_patient & OAI_data.patient_set)
    progression_cohort_images = OAI_data.get_images(patient_id=progression_cohort_patient_6visits,
                                                    part='LEFT_KNEE')

    subcohort_images = progression_cohort_images[16:17]  # 100 patients of progression cohort, 6 visiting each
    analyzer = build_default_analyzer(use_nifty=use_nifti, avsm_path=avsm_path)

    #analyzer.preprocess_parallel(image_list=subcohort_images, n_workers=32, overwrite=False)
    for test_image in subcohort_images:

        # make output path if it does not exist yet
        test_image.create_output_directory(task_name=task_name)

        analyzer.preprocess(test_image, overwrite=overwrite)
        analyzer.segment_image_and_save_results(test_image, overwrite=overwrite)
    analyzer.close_segmenter()

    for i, test_image in enumerate(subcohort_images):
        print("\n[{}] {}\n".format(i, test_image.name))
        analyzer.register_image_to_atlas(test_image, overwrite=overwrite)
        analyzer.extract_surface_mesh(test_image, overwrite=overwrite)
        analyzer.warp_mesh(test_image, overwrite=overwrite, do_clean=do_clean)
        analyzer.eval_registration_surface_distance(test_image)
        analyzer.set_atlas_2D_map(PARAMS['atlas_fc_2d_map_path'], PARAMS['atlas_tc_2d_map_path'])
        analyzer.compute_atlas_2D_map(n_jobs=None)
        analyzer.project_thickness_to_atlas(test_image, overwrite=overwrite)
        analyzer.project_thickness_to_2D(test_image, overwrite=overwrite)
    analyzer.get_surface_distances_eval()

def get_parameter_value(command_line_par,params, params_name, default_val, params_description):

    if command_line_par is None:
        ret = params[(params_name, default_val, params_description)]
    else:
        params[params_name]=(command_line_par, params_description)
        ret = command_line_par

    return ret

def get_parameter_value_flag(command_line_par,params, params_name, default_val, params_description):

    if command_line_par==default_val:
        ret = params[(params_name, default_val, params_description)]
    else:
        params[params_name]=(command_line_par, params_description)
        ret = command_line_par

    return ret

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Performs analysis of the OAI data')

    # specify help strings and default values

    HELP = dict()
    DEFAULT = dict()

    HELP['use_nifty_reg'] = 'If specified (set to True) uses nifty reg to perform registrations, otherwise uses a deep-network based registration.'
    DEFAULT['use_nifty_reg'] = False

    HELP['seed'] = 'Sets the random seed which affects data shuffling.'
    DEFAULT['seed'] = 2018

    HELP['overwrite'] = 'If specified (set to True) overwrites results; otherwise they are not recomputed if they exist'
    DEFAULT['overwrite'] = False

    HELP['config'] = 'The main json configuration file that can be used to define the settings.'
    DEFAULT['config'] = '~/.oai_analysis_settings.json'

    HELP['config_out'] = 'The used json configuration file that the configuration should be written to in the end.'
    DEFAULT['config_out'] = 'tst_out_oai_analysis_settings.json'

    HELP['config_comment_out'] = 'The used json configuration file that the configuration comments should be written to in the end.'
    DEFAULT['config_comment_out'] = 'tst_out_oai_analysis_settings_comments.json'

    HELP['output_directory'] = 'Directory where the analysis results will be stored.'
    DEFAULT['output_directory'] = '/net/biag-raid1/playpen/oai_analysis_results'

    HELP['atlas_image'] = 'Path to the cartilage atlas image; should be called atlas.nii.gz'
    DEFAULT['atlas_image'] = '/playpen/oai/OAI_analysis/atlas/atlas_60_LEFT_baseline_NMI/atlas.nii.gz'

    HELP['oai_data_directory'] = 'Directory where the OAI data can be found.'
    DEFAULT['oai_data_directory'] = '/net/biag-raid/playpen/data/OAI'

    HELP['nifty_reg_directory'] = 'Directory where the nifty-reg binaries live (if niftyreg is used).'
    DEFAULT['nifty_reg_directory'] = '/playpen/oai/niftyreg/install/bin'

    HELP['avsm_directory'] = 'Directory which contains the registration_net scripts; should be ... /registration_net'
    DEFAULT['avsm_directory'] = '/playpen/oai/registration_net'

    HELP['data_division_interval'] = 'Specifies how the data is subdivided. E.g., if one wants to run on 4 machines simultaneously, set it to 4.'
    DEFAULT['data_division_interval'] = 1

    HELP['data_division_offset'] = 'Specified index offset for data subdivision, i.e., if you run on 4 machines, these machines should get offsets 0, 1, 2, and 3 respectively.'
    DEFAULT['data_division_offset'] = 0

    # create parser parameters

    parser.add_argument('--use_nifty_reg', action='store_true', help=HELP['use_nifty_reg'])

    parser.add_argument('--seed', required=False, type=int, default=DEFAULT['seed'], help=HELP['seed'])

    parser.add_argument('--overwrite', action='store_true', help=HELP['overwrite'])

    parser.add_argument('--config', required=False, default=DEFAULT['config'], help=HELP['config'])

    parser.add_argument('--config_out', required=False, default=DEFAULT['config_out'], help=HELP['config_out'])

    parser.add_argument('--config_comment_out', required=False, default=DEFAULT['config_comment_out'], help=HELP['config_comment_out'])

    parser.add_argument('--output_directory', required=False, default=DEFAULT['output_directory'], help=HELP['output_directory'])

    parser.add_argument('--atlas_image', required=False, default=DEFAULT['atlas_image'], help=HELP['atlas_image'])

    parser.add_argument('--oai_data_directory', required=False, default=DEFAULT['oai_data_directory'], help=HELP['oai_data_directory'])

    parser.add_argument('--nifty_reg_directory', required=False, default=DEFAULT['nifty_reg_directory'], help=HELP['nifty_reg_directory'])

    parser.add_argument('--avsm_directory', required=False, default=DEFAULT['avsm_directory'], help=HELP['avsm_directory'])

    parser.add_argument('--data_division_interval', required=False, default=DEFAULT['data_division_interval'], type=int, help=HELP['data_division_interval'])

    parser.add_argument('--data_division_offset', required=False, default=DEFAULT['data_division_offset'], type=int, help=HELP['data_division_offset'])

    parser.add_argument('--task_id', required=False, default=None, type=int, help='When running via slurm on a cluster defines the task ID')

    parser.add_argument('--get_number_of_jobs', action='store_true',
                        help='If set no analysis is run, but the program prints the number of jobs to run (i.e., images to analyze). '
                             'This is useful to set the parameters for SLURM cluster runs w/ run_analysis_on_slurm_cluster.sh')

    args = parser.parse_args()

    if args.config is not None:
        # load the configuration
        PARAMS.load_JSON(args.config)

    # associate parser parameters with the PARAMS structure (i.e., use paramerers from file if available, but can be overwritten by command line arguments)

    get_parameter_value(args.seed, params=PARAMS,
                        params_name='seed',
                        default_val=DEFAULT['seed'],
                        params_description=HELP['seed'])

    get_parameter_value(args.avsm_directory, params=PARAMS,
                        params_name='avsm_directory',
                        default_val=DEFAULT['avsm_directory'],
                        params_description=HELP['avsm_directory'])

    get_parameter_value(args.nifty_reg_directory, params=PARAMS,
                        params_name='nifty_reg_directory',
                        default_val=DEFAULT['nifty_reg_directory'],
                        params_description=HELP['nifty_reg_directory'])

    get_parameter_value(args.atlas_image, params=PARAMS,
                        params_name='atlas_image',
                        default_val=DEFAULT['atlas_image'],
                        params_description=HELP['atlas_image'])

    get_parameter_value(args.oai_data_directory, params=PARAMS,
                        params_name='oai_data_directory',
                        default_val=DEFAULT['oai_data_directory'],
                        params_description=HELP['oai_data_directory'])

    get_parameter_value(args.output_directory, params=PARAMS,
                        params_name='output_directory',
                        default_val=DEFAULT['output_directory'],
                        params_description=HELP['output_directory'])

    get_parameter_value_flag(args.use_nifty_reg, params=PARAMS,
                             params_name='use_nifty_reg',
                             default_val=DEFAULT['use_nifty_reg'],
                             params_description=HELP['use_nifty_reg'])

    get_parameter_value_flag(args.overwrite, params=PARAMS,
                                             params_name='overwrite',
                                             default_val=PARAMS['overwrite'],
                                             params_description=HELP['overwrite'])



    if PARAMS['seed'] is not None:
        print('Setting the random seed to {:}'.format(PARAMS['seed']))
        torch.manual_seed(PARAMS['seed'])
        torch.cuda.manual_seed(PARAMS['seed'])
        np.random.seed(PARAMS['seed'])
        random.seed(PARAMS['seed'])

    # those should be specified at the command line, will not be part of the configuration file
    data_division_interval = args.data_division_interval
    data_division_offset = args.data_division_offset
    task_id = args.task_id

    if task_id is not None:
        if (data_division_interval is not None) or (data_division_offset is not None):
            print('WARNING: data_division settings specified but task ID is given; using task ID {}'.format(task_id))
            data_division_interval = None
            data_division_offset = None


    analyze_cohort(use_nifti=PARAMS['use_nifty_reg'],avsm_path=PARAMS['avsm_directory'],overwrite=PARAMS['overwrite'])
