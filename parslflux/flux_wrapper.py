import sys, getopt
import json, flux
import os
import time
import platform

import yaml

from OAI_analysis.registration.registers import NiftyReg

from flux.job import JobspecV1

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
        image_location = images[image_id]['location']
        image_collectfrom = images[image_id]['collectfrom']
        image_uri = images[image_id]['uri']

        if image_uri == get_own_remote_uri():
            images[image_id]['found'] = True
            continue
        else:
            # send a request to the worker
            rimage = h.rpc (b"workermanager.image.get").get()
            if rimage['success'] == True:
                images[image_id]['found'] = True
                images[image_id]['location'] = rimage['location']
                print ('image retrieved', image_id)
            else:
                images[image_id]['found'] = False
                print ('image not retrieved', image_id)

    return images

def job_execute (h): 
    while True:
        r = h.rpc (b"workermanager.workitem.get").get()
        if "empty" in r.keys():
            print ('empty workqueue')
            time.sleep (5)
        elif "pipelinestages" in r.keys():
            print (r)
            pipelinestages = r['pipelinestages'].split(':')

            images = retrieve_images (r, h)

            for image_id in images.keys():

                if images[image_id]['found'] == True:

                    for pipelinestage in  pipelinestages:
                        execution_script = pipelinestagemap[pipelinestage]
            
        else:
            print ('invalid return')
            time.sleep (5)


if __name__ == "__main__":
    #register parsl manager's flux URI first with the workermanager
    print ('hello world')
    load_config_file(sys.argv[1])
    build_default_analyzer()
    '''
    h = flux.Flux()
    print (sys.argv[1], sys.argv[2])
    r = h.rpc (b"workermanager.worker.register", {"parsluri":sys.argv[2], "workerid":sys.argv[1]})
    job_execute (h)
    '''
