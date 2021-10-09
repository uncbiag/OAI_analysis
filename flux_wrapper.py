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


def construct_image (r, h):
    
    image_collectfrom = r['collectfrom']
    image_uri = r['uri']
    image_location = r['inputlocation']
    image_id = r['id']
    image_version = r['version']

    imagedata = []

    imagedata.append (r['data'])

    print (imagedata)

    image_columns = ['Folder', 'ParticipantID', 'StudyDate', 'Barcode', 'StudyDescription', 'SeriesDescription']

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

def retrieve_image (r, h, OAI_data, image, pipelinestage, output_directory):

    task_name = None if config_data['use_nifty_reg'] else 'avsm'

    os.makedirs (output_directory, exist_ok=True)

    OAI_data.set_processed_data_paths_without_creating_image_directories (output_directory,
                                                                          task_name=task_name)

    image.create_output_directory(task_name=task_name)

    image_collectfrom = r['collectfrom']
    image_uri = r['uri']
    image_location = r['inputlocation']
    image_id = r['id']
    image_version = r['version']

    print ('retrieve_image', image_uri, get_own_remote_uri ())

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

@python_app
def execute_workitem (r, h, output_directory, walltime = 1):

    imageid = r['id']
    version = r['version']

    print (datetime.datetime.now(), 'executing image', imageid, 'version', version)

    analyzer = build_default_analyzer ()

    pipelinestages = r['pipelinestages'].split (':')

    OAI_data = construct_image (r, h)

    analysis_image = OAI_data.get_images ()[0]

    starttime = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    r_starttime, r_endtime = retrieve_image (r, h, OAI_data, analysis_image, pipelinestages[0], output_directory)

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

def execute_workitem_single (r, h, output_directory):
    imageid = r['id']
    version = r['version']

    print (datetime.datetime.now(), 'executing image', imageid, 'version', version)

    analyzer = build_default_analyzer ()

    pipelinestages = r['pipelinestages'].split (':')

    OAI_data = construct_image (r, h)

    analysis_image = OAI_data.get_images ()[0]

    task_name = None if config_data['use_nifty_reg'] else 'avsm'

    OAI_data.set_processed_data_paths_without_creating_image_directories (output_directory,
                                                                          task_name=task_name)

    analysis_image.create_output_directory(task_name=task_name)

    starttime = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    for pipelinestage in  pipelinestages:
       execute_pipelinestage (OAI_data, analyzer,
                              pipelinestage, analysis_image)
    endtime = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    h.rpc (b"workermanager.workitem.report",
           {'version' : version,
            'id': imageid,
            'status' : 'SUCCESS',
            'starttime' : str(starttime),
            'endtime' : str(endtime),
            'outputlocation' : str(analysis_image.folder),
            'r_starttime' : '',
            'r_endtime' : ''})
    print (datetime.datetime.now(), 'imageid', imageid, 'version', version, 'report complete')

def job_execute_single (h, output_directory):
    task_name = None if config_data['use_nifty_reg'] else 'avsm'

    output_directory = output_directory + '/' + platform.node().split('.')[0]

    try:
        print (output_directory)
        shutil.rmtree (output_directory, ignore_errors=True)

    except Exception as e:
        print ('directory not there')

    os.makedirs (output_directory, exist_ok=True)

    while True:
        r = h.rpc (b"workermanager.workitem.get").get()
        if "empty" in r.keys():
            #print (os.getcwd(), 'empty workqueue')
            time.sleep (1)
        elif "pipelinestages" in r.keys():
            print ('r')
            execute_workitem_single (r, h, output_directory)
        else:
            print ('invalid return')
            time.sleep (1)


@bash_app
def execute (output_directory, imageid, version, pipelinestages, collectfrom, image_uri, data0, data1, data2, data3, data4, data5, image_location):
    return "python3.6 execution.py {} {} {} {} {} {} {} {} {} {} {} {} {}".format (output_directory, imageid, \
            version, pipelinestages, collectfrom, image_uri, data0, data1, data2, data3, data4, data5, image_location)

futures = {}

def job_execute (h, output_directory):

    task_name = None if config_data['use_nifty_reg'] else 'avsm'

    output_directory = output_directory + '/' + platform.node().split('.')[0]

    try:
        print (output_directory)
        shutil.rmtree (output_directory, ignore_errors=True)

    except Exception as e:
        print ('directory not there')


    while True:
        is_slot_free = False
        while True:
            if len (futures) > 0:
                for future_key in futures.keys ():
                    future_data = futures[future_key]
                    future = future_data[0]
                    if future.done () == True:
                        exception = future.exception(timeout=1)
                        print ('exception', exception)
                        if exception != None:
                            r = future_data[2]
                            imageid = r['id']
                            version = r['version']
                            print ('future', imageid, 'timedout')
                            h.rpc (b"workermanager.workitem.report",
                                   {'version' : version,
                                    'id': imageid,
                                    'status' : 'FAILED',
                                    'starttime' : str (future_data[1]),
                                    'endtime' : str (datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')),
                                    'outputlocation' : ""})
                        del futures[future_key]
                        is_slot_free = True
                        #print (datetime.datetime.now(), 'free slot')
                        break
                if len (futures) < 2:
                    is_slot_free = True
            else:
                is_slot_free = True
            if is_slot_free == True:
                break

        r = h.rpc (b"workermanager.workitem.get").get()
        if "empty" in r.keys():
            #print ('empty workqueue')
            time.sleep (1)
        else:
            op = r['op']
            if op == 'add':
                #print (r, get_own_remote_uri ())
                timeout = double (r['timeout'])
                data = r['data']
                #print (data[0], data[1], data[2], data[3], data[4], data[5])
                #print (output_directory, r['id'], r['version'], r['pipelinestages'], r['collectfrom'], r['uri'], r['inputlocation'])
                #future=execute_workitem (r, h, output_directory, walltime=100000)
                future = execute (str(output_directory), str(r['id']), str(r['version']), str(r['pipelinestages']), \
                                  str(r['collectfrom']), str(r['uri']), str(data[0]), str(data[1]), \
                                  str(data[2]), str(data[3]), str(data[4]), str(data[5]), str(r['inputlocation']))
                futures[r['id']] = [future, datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), r]
                time.sleep (1)
            else:
                print ('invalid return')
                time.sleep (1)


if __name__ == "__main__":
    #register parsl manager's flux URI first with the workermanager
    h = flux.Flux()
    h.rpc (b"FTmanager.resource.register", {"package": "FT", "name":"node", "path":"/mnt/beegfs/ssbehera/OAI_analysis"})
    entity = h.attr_get("entity").decode("utf-8")
    entityvalue = h.attr_get("entityvalue").decode("utf-8")
    load_config_file ('oaiconfig.yml')
    h.rpc (b"workermanager.worker.register", {"workerid":entityvalue, "parsluri":sys.argv[1]})
    h.rpc (b"exception.register.info", {"jobname":"flux", "jobid":entityvalue, "parenturi":sys.argv[1], "selfuri":get_own_remote_uri()})
    parsl.load (worker_config)
    job_execute (h, sys.argv[2])
    #job_execute_single (h, sys.argv[2])
