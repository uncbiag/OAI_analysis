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

import os, sys
import time
import operator

sys.path.append ('/mnt/beegfs/ssbehera/OAI_analysis/')
from parslflux.resources import ResourceManager
from parslflux.pipeline import PipelineManager
from parslflux.taskset import Taskset
from parslflux.input import InputManager2

@bash_app
def app (command):
    print (command)
    return ''+command
@bash_app
def app1 (entity, entityvalue, scheduleruri):
    return "~/local/bin/flux start -o,--setattr=entity={},--setattr=entityvalue={} sh -c 'flux module load pymod --verbose --path=/home/ssbehera/whesl/FTmanager FTmanager; python3.6 flux_wrapper.py {}'".format('TASK', entityvalue, scheduleruri)
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
                    partition='normal',
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
    future = app1 ('TASK', resource.hostname, get_own_remote_uri ())
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

    h.rpc (b"FTmanager.resource.register", {"package": "FT", "name":"node", "path":"/mnt/beegfs/ssbehera/OAI_analysis"})

    r = ResourceManager (resourcefile, availablefile)
    r.parse_resources ()

    r.purge_resources ()

    r.normalize ()

    '''
    r.sort_by_cpu_ratings ()
    r.sort_by_gpu_ratings ()
    r.sort_by_cpucost ()
    r.sort_by_gpucost ()
    '''

    i = InputManager2 (configfile)

    p = PipelineManager(pipelinefile, cost)
    p.parse_pipelines ()

    '''
    p.performance_extrapolate (r)
    '''

    launch_workers (r.get_resources())

    return r, i, p

g_tasksets = []
g_iterations = {}
g_iteration = 0
g_tasksetid = 0


def create_tasksets (rmanager, imanager, pmanager):
    global g_tasksets
    global g_iteration
    global g_tasksetid
    global g_iterations


    if imanager.get_remaining_count () == 0:
        print ('create_taskset (): no more patients left')
        return -1

    while True:

        ### creation of tasksets for one iteration ###
        ps, endofpipeline = pmanager.get_pipelinestage()
        Ti = Taskset(g_tasksetid, ps.resourcetype, g_iteration)
        g_tasksetid += 1

        Ti.add_pipelinestage(ps)

        while True:
            ps, endofpipeline = pmanager.get_pipelinestage()
            if ps.resourcetype != Ti.get_resource_type():
                pmanager.add_back_pipelinestage()
                break
            Ti.add_pipelinestage(ps)

            if endofpipeline == True:
                Ti.set_endofpipeline(True)
                break

        if Ti.get_taskset_len() > 0:
            #if len(g_tasksets) > 0:
            #    prev_ts = g_tasksets[-1]

                #for resource in resources:
                #    Ti.add_input_taskset(prev_ts, rmanager, pmanager, resource)
            #else:
                #for resource in resources:
                #    Ti.add_input(rmanager, imanager, pmanager, resources)

            g_tasksets.append(Ti)

            #Ti.print_data()

        if endofpipeline == True:
            g_iterations[str(g_iteration)] = g_tasksets
            g_iteration += 1
            g_tasksets = []
            break
    return 0

def delete_taskset_queue ():
    global g_iterations
    global g_iteration

    del g_iterations[g_iteration]
    g_iteration -= 1

def get_taskset (taskset_info):
    global g_iterations

    iteration = taskset_info[0]
    taskset_id = taskset_info[1]

    taskset_queue = g_iterations[str(iteration)]

    for taskset in taskset_queue:
        if taskset_id == taskset.tasksetid:
            return taskset

    return None

def taskset_status (rmanager):
    global g_iterations
    global g_iteration

    print ('#########################')

    resources = rmanager.get_resources ()

    for resource in resources:
        current_cpu_taskset = resource.get_current_taskset ('CPU')
        current_gpu_taskset = resource.get_current_taskset ('GPU')

        if current_cpu_taskset != None:
            print ('status ():', resource.id, current_cpu_taskset[0], current_cpu_taskset[1])
            cpu_taskset = get_taskset (current_cpu_taskset)
            cpu_taskset.get_status (resource.id)

        if current_gpu_taskset != None:
            print ('status ():', resource.id, current_gpu_taskset[0], current_gpu_taskset[1])
            gpu_taskset = get_taskset (current_gpu_taskset)
            gpu_taskset.get_status (resource.id)

    print ('#########################')

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

def update_chunksize (taskset, resource_id, rmanager, pmanager):

    print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    resources = rmanager.get_resources ()

    is_complete = True

    for resource in resources:
        if resource.id == resource_id:
            continue
        if taskset.get_complete (resource.id) == True:
            continue
        if resource.id not in taskset.input:
            is_complete = False
            break
        if taskset.input[resource.id]['count'] > 0 and taskset.input[resource.id]['complete'] != taskset.input[resource.id]['count']:
            is_complete = False
            break

    if is_complete == True:
        print ('is_complete', True)
        resource_exec_times = {}
        for resource in resources:
            exec_times = taskset.get_exectimes (resource.id)
            print ('exectimes:', exec_times)
            if len (exec_times) > 0:
                total_time = 0
                for val in exec_times.values():
                    total_time += val
                avg_time = total_time / len(exec_times)

                resource_exec_times[resource.id] = avg_time
                print ('avg. time taken:', pmanager.encode_pipeline_stages(taskset.pipelinestages), resource.id, resource_exec_times[resource.id])
            else:
                print ('skipping', resource.id)

        sorted_exec_times = sorted(resource_exec_times.items(), key=operator.itemgetter(1), reverse=True)

        basetime_key = sorted_exec_times[0][0]

        basetime = sorted_exec_times[0][1]
        base_resource = rmanager.get_resource (basetime_key)
        base_chunksize = base_resource.get_chunksize (taskset.resourcetype, pmanager.encode_pipeline_stages(taskset.pipelinestages))

        print ('base resource', basetime_key, 'time', basetime)

        for resource_key_value in sorted_exec_times:
            resource_key = resource_key_value[0]
            resource_exec_time = resource_key_value[1]
            if resource_exec_time * 1.2 < basetime:
                resoure = rmanager.get_resource (resource_key)
                resource_old_chunksize = resource.get_chunksize (taskset.resourcetype,
                                            pmanager.encode_pipeline_stages(taskset.pipelinestages))
                resource_chunksize = base_chunksize + int(((basetime - resource_exec_time) / basetime) / .2)

                resource.set_chunksize (taskset.resourcetype, pmanager.encode_pipeline_stages(taskset.pipelinestages), resource_chunksize)
                print (resource.id, pmanager.encode_pipeline_stages(taskset.pipelinestages), 'old chunksize', resource_old_chunksize, 'new chunksize', resource_chunksize)
            else:
                print (resource.id, 'within base time range')
    else:
        print ('is_complete', False)

    print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

def is_free (rmanager, imanager, pmanager, resource):
    global g_iterations
    print ('================================')

    print ('is_free ():', resource.id)

    cpu_current_taskset = resource.get_current_taskset ('CPU')
    gpu_current_taskset = resource.get_current_taskset ('GPU')

    print ('is_free ():', cpu_current_taskset, gpu_current_taskset)

    cpu_free = False
    gpu_free = False

    if cpu_current_taskset == None:
        cpu_free = True
    else:
        cpu_taskset_queue = g_iterations[cpu_current_taskset[0]]
        for taskset in cpu_taskset_queue:
            if taskset.tasksetid == cpu_current_taskset[1]:
                if taskset.input[resource.id]['complete'] == taskset.input[resource.id]['count'] and taskset.input[resource.id]['scheduled'] == 0:
                    cpu_free = True
                    if taskset.input[resource.id]['complete'] == taskset.input[resource.id]['chunksize']:
                        print ('is_free (): CPU', resource.id, 'complete')
                        taskset.set_complete (resource.id, True)
                    update_chunksize (taskset, resource.id, rmanager, pmanager)
                    break


    if gpu_current_taskset == None:
        gpu_free = True
    else:
        gpu_taskset_queue = g_iterations[gpu_current_taskset[0]]
        for taskset in gpu_taskset_queue:
            if taskset.tasksetid == gpu_current_taskset[1]:
                if taskset.input[resource.id]['complete'] == taskset.input[resource.id]['count'] and taskset.input[resource.id]['scheduled'] == 0:
                    gpu_free = True
                    if taskset.input[resource.id]['complete'] == taskset.input[resource.id]['chunksize']:
                        print ('is_free (): GPU', resource.id, 'complete')
                        taskset.set_complete (resource.id, True)
                    update_chunksize (taskset, resource.id, rmanager, pmanager)
                    break

    print ('================================')

    return cpu_free, gpu_free

rmanager = None
imanager = None
pmanager= None

def get_resource_manager ():
    return rmanager

def schedule_next_taskset (rmanager, imanager, pmanager, resource, resourcetype):
    global g_iterations
    global g_iteration

    current_taskset = resource.get_main_taskset(resourcetype)

    print ('schedule_next_taskset ():', resourcetype, resource.id, current_taskset)

    start_iteration = resource.main_iteration

    print ('schedule_next_taskset ():', start_iteration)

    while start_iteration < g_iteration:

        next_taskset_queue = g_iterations[str(start_iteration)]

        prev_taskset = None

        for taskset in next_taskset_queue:

            print (start_iteration, taskset.tasksetid)

            # fix is_complete (chunksize doesn't account correctly)
            if resource.id in taskset.input.keys () and taskset.get_complete(resource.id) == True:
                print ('schedule_next_taskset ():', resourcetype, 'taskset already complete')
            else:
                if taskset.get_resource_type() != resourcetype:
                    print ('schedule_next_taskset ():', resourcetype, 'resource type mismatch')
                    if len (taskset.input) > 0:
                        print ('schedule_next_taskset ():', 'continue')
                        prev_taskset = taskset
                        continue
                    else:
                        print ('schedule_next_taskset ():', 'break')
                        break
                elif taskset.get_resource_type() == resourcetype:
                    print ('schedule_next_taskset ():', resourcetype, 'found a matching taskset')
                    if prev_taskset != None:
                        ret = taskset.add_input_taskset (prev_taskset, rmanager, pmanager, resource)
                        if ret == -1:
                            print ('schedule_next_taskset (): empty prev taskset')
                            prev_taskset = taskset
                            continue
                    else:
                        ret = taskset.add_input (rmanager, imanager, pmanager, resource)
                        if ret == -1:
                            print ('schedule_next_taskset (): no more images left')
                            prev_taskset = taskset
                            continue
                    taskset.submit (rmanager, pmanager, [resource.id])
                    return
            prev_taskset = taskset
        start_iteration += 1

    #Otherwise create a taskset

    latest_taskset_queue = g_iterations[str(g_iteration - 1)]

    first_taskset = latest_taskset_queue[0]

    if first_taskset.get_resource_type () != resourcetype:
        print ('schedule_next_taskset ():', resourcetype, 'first taskset type not', resourcetype)
        return

    print ('schedule_next_taskset ():', resourcetype, 'creating new taskset')

    ret = create_tasksets(rmanager, imanager, pmanager)

    if ret == -1:
        print ('schedule_next_taskset ():', resourcetype, 'no more images left')
        return

    latest_taskset_queue = g_iterations[str(g_iteration - 1)]

    first_taskset = latest_taskset_queue[0]

    first_taskset.add_input (rmanager, imanager, pmanager, resource)

    first_taskset.submit (rmanager, pmanager, [resource.id])

def OAI_scheduler (configfile, pipelinefile, resourcefile, availablefile, cost):

    global g_iterations
    global g_iteration
    global rmanager
    global imanager
    global pmanager

    rmanager, imanager, pmanager = setup (resourcefile, pipelinefile, configfile, availablefile)

    print ('OAI_scheduler ():', 'sleeping for 40 secs')

    time.sleep (40)

    create_tasksets(rmanager, imanager, pmanager)

    tasksets = g_iterations['0']

    resources = rmanager.get_resources()

    for resource in resources:
        tasksets[0].add_input (rmanager, imanager, pmanager, resource)

    tasksets[0].submit (rmanager, pmanager, [])

    for resource in resources:
        resource.main_iteration = g_iteration - 1

    while True:

        taskset_status (rmanager)

        for resource in resources:
            cpu_free, gpu_free = is_free (rmanager, imanager, pmanager, resource)

            cpu_current_taskset = resource.get_current_taskset ('CPU')
            gpu_current_taskset = resource.get_current_taskset ('GPU')

            print (cpu_free, gpu_free)
            print (cpu_current_taskset, gpu_current_taskset)

            if cpu_free:
                schedule_next_taskset (rmanager, imanager, pmanager, resource, 'CPU')
            if gpu_free:
                schedule_next_taskset (rmanager, imanager, pmanager, resource, 'GPU')

        print ('OAI_scheduler ():', 'sleeping for 5')

        time.sleep (5)

        worker_status ()

if __name__ == "__main__":

    configfile = sys.argv[1]

    pipelinefile = sys.argv[2]

    resourcefile = sys.argv[3]

    availablefile = sys.argv[4]

    cost = float (sys.argv[5])

    OAI_scheduler (configfile, pipelinefile, resourcefile, availablefile, cost)
