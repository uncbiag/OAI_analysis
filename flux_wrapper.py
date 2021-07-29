import sys, getopt
import json, flux
import os
import time
import platform

import yaml
import pandas as pd

from datetime import datetime
from registration.registers import NiftyReg
from segmentation.segmenter import Segmenter3DInPatchClassWise
from oai_image_analysis import OAIImageAnalysis
from data.OAI_data import OAIData, OAIImage, OAIPatients

from flux.job import JobspecV1

from parsl.config import Config
from parsl.executors import ThreadPoolExecutor
from parsl import python_app, bash_app
import parsl

worker_config = Config (
    executors = [
        ThreadPoolExecutor(
            label='worker_thread'
        )
    ]
)

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

pipelinestagemap = {'preprocess': './flux/preprocess.py',
                    'segmentation': './flux/segmentation.py',
                    'extractsurfacemesh': './flux/extractsurfacemesh.py',
                    'registerimagetoatlas' : './flux/registerimgtoatlas.py',
                    'warpmesh': './flux/warpmesh.py',
                    'evalregistration': './flux/evalregsurfacedist.py',
                    'setatlas2dmap': './flux/setatlas2dmap.py',
                    'computeatlas2dmap': './flux/computeatlas2dmap.py',
                    'projectthicknesstoatlas': './flux/projectthicknesstoatlas.py',
                    'projectthicknessto2d': './flux/projectthicknessto2D.py'}

def get_own_remote_uri():
    localuri = os.getenv('FLUX_URI')
    remoteuri = localuri.replace("local://", "ssh://" + platform.node().split('.')[0])
    return remoteuri

def retrieve_images (r, h):
    images = r['images']

    for image_id in images.keys():
        image_collectfrom = images[image_id]['collectfrom']
        image_uri = images[image_id]['uri']

        if image_uri == get_own_remote_uri():
            images[image_id]['found'] = True
            continue
        else:
            # send a request to the worker
            rimage = h.rpc (b"workermanager.image.get",
                            {'iteration':r['iteration'],
                            'tasksetid':r['tasksetid'], 'imageid':image_id}).get()
            if rimage['success'] == True:
                images[image_id]['found'] = True
                print ('image retrieved', image_id)
            else:
                images[image_id]['found'] = False
                print ('image not retrieved', image_id)

    imagedata = []

    for image_id in images.keys():
        if images[image_id]['found'] == True:
            imagedata.append (images[image_id]['data'])

    print (imagedata)

    image_columns = ['Folder', 'ParticipantID', 'StudyDate', 'Barcode', 'StudyDescription', 'SeriesDescription']

    image_df = pd.DataFrame (imagedata, columns = image_columns)

    platform_name = platform.node().split('.')[0]

    image_data_sheet = image_df.to_csv (platform_name + '.csv')

    image_data_directory = config_data['oai_data_directory']

    OAI_data = OAIData (platform_name + '.csv', image_data_directory)

    os.remove (platform_name + '.csv')

    return OAI_data

def preprocess (analyzer, image):
    task_name = None if config_data['use_nifty_reg'] else 'avsm'

    print("[{}] Preprocess".format(str(image)))
    try:
        image.create_output_directory(task_name=task_name)
        analyzer.preprocess (image, overwrite=config_data['overwrite'])
    except Exception as e:
        err_msg = 'Could not preprocess image: {}'.format (str(image))
        print (err_msg)
        print (e)

def segmentation (analyzer, image):
    print ("[{}] Segmentation".format(str(image)))
    try:
        analyzer.segment_image_and_save_results(image, overwrite=config_data['overwrite'])
    except Exception as e:
        err_msg = 'Could not segment image: {}'.format (str(image))
        print (err_msg)
        print (e)

def extract_surface_mesh (analyzer, image):
    print ("[{}] Extract surface mesh".format(str(image)))

    try:
        analyzer.extract_surface_mesh (image, overwrite = config_data['overwrite'])
    except Exception as e:
        err_msg = 'Could not extract surface mesh {}'.format (str(image))
        print (err_msg)
        print (e)

def register_image_to_atlas (analyzer, image):
    print ("[{}] Register image to atlas".format (str(image)))

    try:
        analyzer.register_image_to_atlas (image, overwrite = config_data['overwrite'])
    except Exception as e:
        err_msg = 'Could not register image to atlas {}'.format (str(image))
        print (err_msg)
        print (e)

def warp_mesh (analyzer, image):
    print ("[{}] Warp Mesh".format (str(image)))

    try:
        analyzer.warp_mesh (image, overwrite = config_data['overwrite'], do_clean = False)
    except Exception as e:
        err_msg = 'Could not warp mesh {}'.format (str(image))
        print (err_msg)
        print (e)

def eval_registration_surface_distance (analyzer, image):
    print ("[{}] Eval registration sufrace distance".format (str(image)))

    try:
        analyzer.eval_registration_surface_distance (image)
    except Exception as e:
        err_msg = 'Could not eval registration {}'.format (str(image))
        print (err_msg)
        print (e)

    print ("[{}] set atlas 2d map".format (str(image)))

    try:
        analyzer.set_atlas_2D_map (config_data['atlas_fc_2d_map_path'],
                                   config_data['atlas_tc_2d_map_path'])

    except Exception as e:
        err_msg = 'Could not set atlas 2d map {}'.format (str(image))
        print (err_msg)
        print (e)

    print ("[{}] compute atlas 2D map".format (str(image)))

    try:
        analyzer.compute_atlas_2D_map (n_jobs = None)

    except Exception as e:
        err_msg = 'Could not compute atlas 2D map'.format (str(image))
        print (err_msg)
        print (e)

def project_thickness_to_atlas (analyzer, image):
    print ("[{}] project thickness to atlas".format (str(image)))

    try:
        analyzer.project_thickness_to_atlas (image, overwrite = config_data['overwrite'])

    except Exception as e:
        err_msg = 'Could not project thickness to atlas'.format (str(image))
        print (err_msg)
        print (e)


    print ("[{}] project thickness to 2D".format (str(image)))

    try:
        analyzer.project_thickness_to_2D (image, overwrite = config_data['overwrite'])

    except Exception as e:
        err_msg = 'Could not project thickness to 2D'.format (str(image))
        print (err_msg)
        print (e)

    print ("[{}] get surface distances eval".format (str(image)))

    try:
        analyzer.get_surface_distances_eval ()

    except Exception as e:
        err_msg = 'Could not get surface distances evaluation'.format ((image))
        print (err_msg)
        print (e)

def execute_pipelinestage (OAI_data, analyzer, pipelinestage, image):
    task_name = None if config_data['use_nifty_reg'] else 'avsm'

    OAI_data.set_processed_data_paths_without_creating_image_directories(config_data['output_directory'],
                                                                         task_name=task_name)
    start_time = time.time ()

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

    end_time = time.time ()

    return end_time - start_time

@python_app
def execute_workitem (r, h):
    tasksetid = r['tasksetid']
    iteration = r['iteration']

    print (datetime.now(), 'executing iteration', iteration, 'tasksetid', tasksetid)

    analyzer = build_default_analyzer ()

    pipelinestages = r['pipelinestages'].split (':')

    OAI_data = retrieve_images (r, h)

    analysis_images = OAI_data.get_images ()

    time_stats = []

    for i, image in enumerate (analysis_images):
       for pipelinestage in  pipelinestages:
           time_taken = execute_pipelinestage (OAI_data, analyzer,
                                               pipelinestage,
                                               image)
           print (datetime.now(), 'iteration', iteration, 'tasksetid', tasksetid,
                  image.name, pipelinestage, 'time taken', time_taken)
           time_stats.append (str(time_taken))

    print (datetime.now(), 'reporting', iteration, tasksetid, time_stats)

    h.rpc (b"workermanager.workitem.report",
           {'tasksetid' : tasksetid,
            'iteration' : iteration,
            'status' : 'SUCCESS',
            'times' : time_stats})
    print (datetime.now(), 'iteration', iteration, 'tasksetid', tasksetid, 'report complete')

futures = []

def job_execute (h): 
    while True:
        is_slot_free = False
        while True:
            if len (futures) > 1:
                for future in futures:
                    if future.done () == True:
                        futures.remove (future)
                        is_slot_free = True
                        print (datetime.now(), 'free slot')
                        break
            else:
                is_slot_free = True
            if is_slot_free == True:
                break

        r = h.rpc (b"workermanager.workitem.get").get()
        if "empty" in r.keys():
            print ('empty workqueue')
            time.sleep (5)
        elif "pipelinestages" in r.keys():
            print (r)
            future = execute_workitem (r, h)
            futures.append (future)
            time.sleep (5)
        else:
            print ('invalid return')
            time.sleep (5)


if __name__ == "__main__":
    #register parsl manager's flux URI first with the workermanager
    h = flux.Flux()
    print (h.attr_get("entity"), h.attr_get("entityvalue"))
    print (sys.argv[1], sys.argv[2])
    load_config_file ('oaiconfig.yml')
    h.rpc (b"workermanager.worker.register", {"workerid":sys.argv[1], "parsluri":sys.argv[2]})
    h.rpc (b"exception.register.info", {"jobname":"flux", "jobid":sys.argv[1], "parenturi":sys.argv[2], "selfuri":get_own_remote_uri()})
    parsl.load (worker_config)
    job_execute (h)
