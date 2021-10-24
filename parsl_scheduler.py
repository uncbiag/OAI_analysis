from parsl.config import Config
from parsl.channels import SSHChannel
from parsl.channels import LocalChannel
from parsl.providers import SlurmProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_query
from parsl import python_app, bash_app
import parsl
import json, flux
import platform
import datetime
import os, sys
import time
import operator

sys.path.append ('/mnt/beegfs/ssbehera/OAI_analysis/')
from parslflux.resources import ResourceManager
from parslflux.pipeline import PipelineManager
from parslflux.taskset import Taskset
from parslflux.input import InputManager2
from parslflux.scheduling_policy import Policy

from parslflux.FirstCompleteFirstServe import FirstCompleteFirstServe
from parslflux.FastCompleteFirstServe import FastCompleteFirstServe
from parslflux.FastCompleteFirstServe2 import FastCompleteFirstServe2
from parslflux.FastCompleteFirstServe3 import FastCompleteFirstServe3
from parslflux.FastCompleteFirstServe4 import FastCompleteFirstServe4
from parslflux.FastCompleteFirstServe5 import FastCompleteFirstServe5
from parslflux.FastCompleteFirstServe6 import FastCompleteFirstServe6
from parslflux.FastCompleteFirstServe5Alloc import FastCompleteFirstServe5Alloc
from parslflux.FastCompleteFirstServe6Alloc import FastCompleteFirstServe6Alloc
from parslflux.FastCompleteFirstServe7Alloc import FastCompleteFirstServe7Alloc
from parslflux.DFS import DFS

@bash_app
def app (command):
    print (command)
    return ''+command
@bash_app
def app1 (entity, entityvalue, scheduleruri, output_location):
    return "~/local/bin/flux start -o,--setattr=entity={},--setattr=entityvalue={} sh -c 'flux module load pymod --verbose --path=/home/ssbehera/whesl/FTmanager FTmanager; python3.6 flux_wrapper.py {} {}'".format('TASK', entityvalue, scheduleruri, output_location)
    #return '~/local/bin/flux start -o,--setattr=entity={},--setattr=entityvalue={} flux module load pymod --verbose --path=/home/ssbehera/whesl/FTmanager FTmanager python3.6 flux_wrapper.py {}'.format ('TASK', entityvalue, scheduleruri)
    #return '~/local/bin/flux start -o,--setattr=entity={},--setattr=entityvalue={}; flux module load pymod --verobse --path=/home/ssbehera/whesl/FTmanager FTmanager; python3.6 /home/ssbehera/whesl/whesl.py handler FT node /mnt/beegfs/ssbehera/OAI_analysis;  python3.6 flux_wrapper.py {}'.format('TASK', entityvalue, scheduleruri)

def get_own_remote_uri():
    localuri = os.getenv('FLUX_URI')
    remoteuri = localuri.replace("local://", "ssh://" + platform.node().split('.')[0])
    return remoteuri

def get_launch_config (options):
    inspector_config = Config (
        executors=[
            HighThroughputExecutor(
                label="inspector",
                address=str(platform.node().split('.')[0]),
                max_workers=1,
                provider=SlurmProvider(
                    channel=LocalChannel(),
                    nodes_per_block=1,
                    min_blocks=1,
                    max_blocks=1,
                    init_blocks=1,
                    partition='max',
                    walltime='24:00:00',
                    scheduler_options=options,
                    launcher=SrunLauncher(),
                ),
            )
        ],
        app_cache=False,
    )
    return inspector_config

def launch_worker (resource):
    parsl.clear()
    options = '#SBATCH -w ' + resource.hostname
    config = get_launch_config (options)
    parsl.load (config)
    future = app1 ('TASK', resource.hostname, get_own_remote_uri (), resource.output_location)
    #app ('~/local/bin/flux start -o,--setattr=entity={},--setattr=entityvalue={} python3.6 flux_wrapper.py {} {}'.format('TASK', resource.hostname, resource.hostname, get_own_remote_uri()))
    print ('launched', resource.hostname)
    return future, 'TASK', resource.hostname

worker_futures = {}

def launch_workers (resources):
    for resource in resources:
        ret = launch_worker (resource)
        worker_futures[str (resource.id)] = ret
        time.sleep (5) #add sleep to avoid same exec dir

def setup (resourcefile, pipelinefile, configfile, availablefile):

    h = flux.Flux()

    h.rpc (b"FTmanager.resource.register", {"package": "FT", "name":"node", "path":"/mnt/beegfs/ssbehera/OAI_analysis", "pid":os.getpid()})

    r = ResourceManager (resourcefile, availablefile)

    print ('1')

    r.parse_resources ()

    print ('2')

    r.purge_resources ()

    print ('3')

    i = InputManager2 (configfile)

    p = PipelineManager(pipelinefile, cost)

    p.parse_pipelines ()

    launch_workers (r.get_resources())

    return r, i, p

def worker_status ():
    print ('*************************')
    for worker in worker_futures.keys ():
        future = worker_futures[worker][0]
        if future.task_status() == 'failed':
            #raise an exception
            print (worker, 'failed')
            entity = worker_futures[worker][1]
            entityvalue = worker_futures[worker][2]
            h = flux.Flux()
            ownentity = h.attr_get("entity").decode("utf-8") 
            ownentityvalue = h.attr_get("entityvalue").decode("utf-8")
            h.rpc (b"exception.catch", {"exception":"nodefailure", "fromentity":ownentity, "fromentityvalue":ownentityvalue, "originentity":entity, "originentityvalue":entityvalue})
    print ('*************************')

rmanager = None
imanager = None
pmanager= None

def get_resource_manager ():
    return rmanager


def DFS_scheduler (configfile, pipelinefile, resourcefile, availablefile, cost):
    global rmanager, imanager, pmanager

    rmanager, imanager, pmanager = setup (resourcefile, pipelinefile, configfile, availablefile)

    print ('DFS_scheduler ()', 'waiting for 40 secs')

    time.sleep (40)

    scheduling_policy = DFS ("DFS")

    while True:
        resources = rmanager.get_resources ()

        for resource in resources:
            resource.get_status_whole (pmanager)

            scheduling_policy.remove_complete_workitem_whole (resource)

        empty_resources = []

        for resource in resources:
            empty = resource.is_empty_whole ()

            if empty == True:
                empty_resources.append (resource)

        #print (empty_resources)

        if len (empty_resources) > 0:
            scheduling_policy.add_new_workitems (rmanager, imanager, pmanager, empty_resources)

        idle_resources = []

        for resource in resources:
            idle = resource.is_idle_whole ()

            if idle == True:
                idle_resources.append (resource)

        for idle_resource in idle_resources:
            #print ('scheduling cpu', idle_cpu.id)
            idle_resource.schedule_whole (rmanager, pmanager)

        idle_resources = []

        for resource in resources:
            idle = resource.is_idle_whole ()

            if idle == True:
                idle_resources.append (resource)

        if len (idle_resources) == len (resources):
            print ('all tasks complete')
            break

        time.sleep (1)

def OAI_scheduler_2 (configfile, pipelinefile, resourcefile, availablefile, cost):
    global rmanager, imanager, pmanager

    rmanager, imanager, pmanager = setup (resourcefile, pipelinefile, configfile, availablefile)

    print ('OAI_scheduler_2 ()', 'waiting for 40 secs')

    time.sleep (40)

    scheduling_policy = FirstCompleteFirstServe("First-0")
    #scheduling_policy = FastCompleteFirstServe("Fast-1")
    #scheduling_policy = FastCompleteFirstServe2("Fast-2")
    #scheduling_policy = FastCompleteFirstServe3("Fast-3")
    #scheduling_policy = FastCompleteFirstServe4("Fast-4")
    #scheduling_policy = FastCompleteFirstServe5("Fast-5")
    #scheduling_policy = FastCompleteFirstServe6("Fast-6")
    #scheduling_policy = FastCompleteFirstServe5Alloc("Fast-5-Alloc")
    #scheduling_policy = FastCompleteFirstServe6Alloc("Fast-6-Alloc")
    #scheduling_policy = FastCompleteFirstServe7Alloc("Fast-7-Alloc")

    while True:

        resources = rmanager.get_resources ()

        for resource in resources:
            #print ('###########################')
            resource.get_status (pmanager)
            #print ('###########################')
            #print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            scheduling_policy.remove_complete_workitem (resource)
            #print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        empty_cpus = []
        empty_gpus = []

        for resource in resources:
            #print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            cpu_empty, gpu_empty = resource.is_empty ()
            #print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

            if cpu_empty == True:
                empty_cpus.append (resource)
            if gpu_empty == True:
                empty_gpus.append (resource)

        if len (empty_cpus) > 0:
            #print ('****************************')
            scheduling_policy.add_new_workitems (rmanager, imanager, pmanager, empty_cpus, 'CPU')
            #print ('****************************')
        if len (empty_gpus) > 0:
            #print ('&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            scheduling_policy.add_new_workitems (rmanager, imanager, pmanager, empty_gpus, 'GPU')
            #print ('&&&&&&&&&&&&&&&&&&&&&&&&&&&&')


        idle_cpus = []
        idle_gpus = []

        #print ('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        for resource in resources:
            cpu_idle, gpu_idle = resource.is_idle ()

            if cpu_idle == True:
                idle_cpus.append (resource)
            if gpu_idle == True:
                idle_gpus.append (resource)

        #print (idle_cpus)
        #print (idle_gpus)
        #print ('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        for idle_cpu in idle_cpus:
            #print ('scheduling cpu', idle_cpu.id)
            idle_cpu.schedule (rmanager, pmanager, 'CPU')

        for idle_gpu in idle_gpus:
            #print ('scheduling gpu', idle_gpu.id)
            idle_gpu.schedule (rmanager, pmanager, 'GPU')

        idle_cpus = []
        idle_gpus = []

        for resource in resources:
            cpu_idle, gpu_idle = resource.is_idle ()

            if cpu_idle == True:
                idle_cpus.append (resource)
            if gpu_idle == True:
                idle_gpus.append (resource)

        if len (idle_cpus) == rmanager.get_cpu_resources_count () and len (idle_gpus) == rmanager.get_gpu_resources_count ():
            print ('all tasks complete')
            break

        time.sleep (1)

if __name__ == "__main__":

    configfile = sys.argv[1]

    pipelinefile = sys.argv[2]

    resourcefile = sys.argv[3]

    availablefile = sys.argv[4]

    cost = float (sys.argv[5])

    #OAI_scheduler_2 (configfile, pipelinefile, resourcefile, availablefile, cost)

    DFS_scheduler (configfile, pipelinefile, resourcefile, availablefile, cost)
