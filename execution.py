import sys, getopt
import json, flux
import os
import time
import platform

import yaml
import pandas as pd

import datetime
from registration.registers import NiftyReg, AVSMReg
from segmentation.segmenter import Segmenter3DInPatchClassWise
from oai_image_analysis import OAIImageAnalysis
from data.OAI_data import OAIData, OAIImage, OAIPatients

from flux.job import JobspecV1

from parsl.config import Config
from parsl.executors import ThreadPoolExecutor
from parsl import python_app, bash_app
import parsl
from parsl.app.errors import AppTimeout
import subprocess
from numpy import double
import shutil

config_data = {}

def load_config_file(config_file):
    global config_data
    config_data_file = open (config_file)
    config_data = yaml.load (config_data_file, Loader=yaml.FullLoader)


def build_default_analyzer(ckpoint_folder=None, use_nifty=True,avsm_path=None):
    niftyreg_path = config_data['nifty_reg_directory']
    avsm_path = config_data['avsm_directory']
    avsm_path = avsm_path + '/demo'
    use_nifty = config_data['use_nifty_reg']


    register = NiftyReg(niftyreg_path) if use_nifty else AVSMReg(avsm_path=avsm_path,python_executable=config_data['python_executable'])
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
    analyzer.set_atlas(atlas_image_file=config_data['atlas_image'], atlas_FC_mesh_file=config_data['atlas_fc_mesh_path'],
                       atlas_TC_mesh_file=config_data['atlas_tc_mesh_path'])
    analyzer.set_register(register=register, affine_config=affine_config, bspline_config=bspline_config)
    analyzer.set_segmenter(segmenter=segmenter)
    analyzer.set_preprocess(bias_correct=False, reset_coord=True, normalize_intensity=True, flip_to="LEFT")
    return analyzer

def get_own_remote_uri():
    localuri = os.getenv('FLUX_URI')
    remoteuri = localuri.replace("local://", "ssh://" + platform.node().split('.')[0])
    return remoteuri


def construct_image (image_id, image_version, data):
    image_columns = ['Folder', 'ParticipantID', 'StudyDate', 'Barcode', 'StudyDescription', 'SeriesDescription']

    imagedata = []
    imagedata.append (data)

    image_df = pd.DataFrame (imagedata, columns = image_columns)

    platform_name = platform.node().split('.')[0]

    image_data_sheet = image_df.to_csv (platform_name + str(image_id) + str(image_version) + '.csv')

    image_data_directory = config_data['oai_data_directory']

    OAI_data = OAIData (platform_name + str(image_id) + str(image_version) + '.csv', image_data_directory)

    os.remove (platform_name + str(image_id) + str(image_version) + '.csv')

    return OAI_data

def preprocess (analyzer, image):

    print("[{}] Preprocess".format(str(image)))
    try:
        analyzer.preprocess (image, overwrite=config_data['overwrite'])
    except Exception as e:
        err_msg = 'Could not preprocess image: {}'.format (str(image))
        print (err_msg)
        print (e)
        raise Exception('Failure') from e

def segmentation (analyzer, image):
    print ("[{}] Segmentation".format(str(image)))
    try:
        analyzer.segment_image_and_save_results(image, overwrite=config_data['overwrite'])
    except Exception as e:
        err_msg = 'Could not segment image: {}'.format (str(image))
        print (err_msg)
        print (e)
        raise Exception('Failure') from e

def extract_surface_mesh (analyzer, image):
    print ("[{}] Extract surface mesh".format(str(image)))

    try:
        analyzer.extract_surface_mesh (image, overwrite = config_data['overwrite'])
    except Exception as e:
        err_msg = 'Could not extract surface mesh {}'.format (str(image))
        print (err_msg)
        print (e)
        raise Exception('Failure') from e

def register_image_to_atlas (analyzer, image):
    print ("[{}] Register image to atlas".format (str(image)))

    try:
        analyzer.register_image_to_atlas (image, overwrite = config_data['overwrite'])
    except Exception as e:
        err_msg = 'Could not register image to atlas {}'.format (str(image))
        print (err_msg)
        print (e)
        raise Exception('Failure') from e

def warp_mesh (analyzer, image):
    print ("[{}] Warp Mesh".format (str(image)))

    try:
        analyzer.warp_mesh (image, overwrite = config_data['overwrite'], do_clean = False)
    except Exception as e:
        err_msg = 'Could not warp mesh {}'.format (str(image))
        print (err_msg)
        print (e)
        raise Exception('Failure') from e

def eval_registration_surface_distance (analyzer, image):
    print ("[{}] Eval registration sufrace distance".format (str(image)))

    try:
        analyzer.eval_registration_surface_distance (image)
    except Exception as e:
        err_msg = 'Could not eval registration {}'.format (str(image))
        print (err_msg)
        print (e)
        raise Exception('Failure') from e

    print ("[{}] set atlas 2d map".format (str(image)))

    try:
        analyzer.set_atlas_2D_map (config_data['atlas_fc_2d_map_path'],
                                   config_data['atlas_tc_2d_map_path'])

    except Exception as e:
        err_msg = 'Could not set atlas 2d map {}'.format (str(image))
        print (err_msg)
        print (e)
        raise Exception('Failure') from e

    print ("[{}] compute atlas 2D map".format (str(image)))

    try:
        analyzer.compute_atlas_2D_map (n_jobs = None)

    except Exception as e:
        err_msg = 'Could not compute atlas 2D map'.format (str(image))
        print (err_msg)
        print (e)
        raise Exception('Failure') from e

def project_thickness_to_atlas (analyzer, image):
    print ("[{}] project thickness to atlas".format (str(image)))

    try:
        analyzer.project_thickness_to_atlas (image, overwrite = config_data['overwrite'])

    except Exception as e:
        err_msg = 'Could not project thickness to atlas'.format (str(image))
        print (err_msg)
        print (e)
        raise Exception('Failure') from e

    print ("[{}] project thickness to 2D".format (str(image)))

    try:
        analyzer.project_thickness_to_2D (image, overwrite = config_data['overwrite'])

    except Exception as e:
        err_msg = 'Could not project thickness to 2D'.format (str(image))
        print (err_msg)
        print (e)
        raise Exception('Failure') from e

    print ("[{}] get surface distances eval".format (str(image)))

    try:
        analyzer.get_surface_distances_eval ()

    except Exception as e:
        err_msg = 'Could not get surface distances evaluation'.format ((image))
        print (err_msg)
        print (e)
        raise Exception('Failure') from e

def execute_pipelinestage (OAI_data, analyzer, pipelinestage, image):
    try:
        if pipelinestage == 'preprocess':
            preprocess (analyzer, image)
        elif pipelinestage == 'segmentation':
            segmentation (analyzer, image)
        elif pipelinestage == 'extractsurfacemesh':
            extract_surface_mesh (analyzer, image)
        elif pipelinestage == 'registerimagetoatlas':
            register_image_to_atlas (analyzer, image)
        elif pipelinestage == 'warpmesh':
            warp_mesh (analyzer, image)
        elif pipelinestage == 'evalregistration':
            eval_registration_surface_distance (analyzer, image)
        elif pipelinestage == 'projectthicknesstoatlas':
            project_thickness_to_atlas (analyzer, image)
    except Exception as e:
        raise Exception ('Failure') from e


def retrieve_image (image_id, image_version, pipelinestage, image_collectfrom, \
                    image_uri, image_location, image, output_directory, OAI_data):

    task_name = None if config_data['use_nifty_reg'] else 'avsm'

    os.makedirs (output_directory, exist_ok=True)

    OAI_data.set_processed_data_paths_without_creating_image_directories (output_directory,
                                                                          task_name=task_name)

    image.create_output_directory(task_name=task_name)

    if image_uri == get_own_remote_uri ():
        return '', ''
    else:
        #copy the file to right location
        starttime = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        srcs = []
        dests = []
        if pipelinestage == 'preprocess':
            pass
        elif pipelinestage == 'segmentation':
            srcs.append (os.path.join (image_location, 'image_preprocessed.nii.gz'))
            dests.append (image.folder)
        elif pipelinestage == 'extractsurfacemesh':
            srcs.append (os.path.join (image_location, 'image_preprocessed.nii.gz'))
            srcs.append (os.path.join (image_location, 'FC_probmap.nii.gz'))
            srcs.append (os.path.join (image_location, 'TC_probmap.nii.gz'))
            dests.append (image.folder)
            dests.append (image.folder)
            dests.append (image.folder)
        elif pipelinestage == 'registerimagetoatlas':
            srcs.append (os.path.join (image_location, 'image_preprocessed.nii.gz'))
            srcs.append (os.path.join(image_location, task_name, "FC_mesh_world.ply"))
            srcs.append (os.path.join(image_location, task_name, "TC_mesh_world.ply"))
            
            dests.append (image.folder)
            dests.append (os.path.join(image.folder, task_name))
            dests.append (os.path.join(image.folder, task_name))
        elif pipelinestage == 'warpmesh':
            srcs.append (os.path.join(image_location, 'image_preprocessed.nii.gz'))
            srcs.append (os.path.join(image_location, task_name, "FC_mesh_world.ply"))
            srcs.append (os.path.join(image_location, task_name, "TC_mesh_world.ply"))
            srcs.append (os.path.join(image_location, task_name, "inv_transform_to_atlas.nii.gz"))

            dests.append (image.folder)
            dests.append (os.path.join(image.folder, task_name))
            dests.append (os.path.join(image.folder, task_name))
            dests.append (os.path.join(image.folder, task_name))

        print ('retrieve_image', srcs, dests)

        index = 0
        for src in srcs:
            print ("scp", image_collectfrom + ':' + srcs[index], dests[index])
            subprocess.run(["scp", image_collectfrom + ':' + srcs[index], dests[index]])
            index += 1

        endtime = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        return starttime, endtime

if __name__ == "__main__":
    #register parsl manager's flux URI first with the workermanager
    load_config_file ('oaiconfig.yml')

    h = flux.Flux()
    output_directory = sys.argv[1]
    imageid = sys.argv[2]
    version = sys.argv[3]
    pipelinestages = sys.argv[4]
    collectfrom = sys.argv[5]
    image_uri = sys.argv[6]
    data0 = sys.argv[7]
    data1 = sys.argv[8]
    data2 = sys.argv[9]
    data3 = sys.argv[10]
    data4 = sys.argv[11] + ' ' + sys.argv[12]
    data5 = sys.argv[13]

    if len (sys.argv) == 14:
        image_location = ''
    else:
        image_location = sys.argv[14]

    print (sys.argv)

    data = [data0, data1, data2, data3, data4, data5]

    print (data)

    analyzer = build_default_analyzer ()

    pipelinestages = pipelinestages.split (':')

    OAI_data = construct_image (imageid, version, data)

    analysis_image = OAI_data.get_images ()[0]

    starttime = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    r_starttime, r_endtime = retrieve_image (imageid, version, pipelinestages[0], collectfrom, \
                                image_uri, image_location, analysis_image, output_directory, OAI_data)

    try:
        for pipelinestage in  pipelinestages:
           execute_pipelinestage (OAI_data, analyzer,
                                  pipelinestage, analysis_image)
        endtime = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        raise Exception ('Failure') from e

    h.rpc (b"workermanager.workitem.report",
           {'version' : version,
            'id': imageid,
            'status' : 'SUCCESS',
            'starttime' : str(starttime),
            'endtime' : str(endtime),
            'r_starttime' : str(r_starttime),
            'r_endtime' : str(r_endtime),
            'outputlocation' : str(analysis_image.folder)})
    print (datetime.datetime.now(), 'imageid', imageid, 'version', version, 'report complete')
