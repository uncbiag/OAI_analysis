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
    return '~/local/bin/flux start -o,--setattr=entity={},--setattr=entityvalue={} python3.6 flux_wrapper.py {}'.format('TASK', entityvalue, scheduleruri)

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
    app1 ('TASK', resource.hostname, get_own_remote_uri ())
    #app ('~/local/bin/flux start -o,--setattr=entity={},--setattr=entityvalue={} python3.6 flux_wrapper.py {} {}'.format('TASK', resource.hostname, resource.hostname, get_own_remote_uri()))
    print ('launched', resource.hostname)

def launch_workers (resources):
    for resource in resources:
        launch_worker (resource)

def setup (resourcefile, pipelinefile, configfile, availablefile):
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

def update_chunks (rmanager, pmanager, tasksets):
    time.sleep(5)
    #send a request to parslmanager to get updates
    for taskset in tasksets:
        total_latency = 0
        for key in taskset.input.keys():
            mapping = taskset.input[key]
            resourceid = key
            resource_latency = rmanager.get_latency (resourceid, taskset.pipelinestages) #get latency through
            if resource_latency == -1:
                print (resourceid, 'taskset not complete yet')
            else:
                total_latency += resource_latency
                mapping['complete'] = True

        for key in taskset.input.keys():
            mapping = taskset.input[key]

            print('count: ', mapping['count'])
            print('images:')
            images = mapping['images']
            for key1 in images.keys():
                image = images[key1]
                print('id: ', key1, 'name: ', image['name'], 'location: ', image['location'], 'collectfrom: ',
                      image['collectfrom'])
                print('-------------------')

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

def schedule_taskset_main (rmanager, imanager, pmanager, resource_id, main_taskset_info):
    global g_iterations
    global g_iteration

    print ('schedule_taskset_main ():')

    resource = rmanager.get_resource(resource_id)

    main_taskset_iteration = main_taskset_info[0]

    new_taskset = None

    main_taskset = get_taskset (main_taskset_info)

    if main_taskset.get_endofpipeline () == True:
        if resource.main_iteration == g_iteration - 1:
            ret = create_tasksets (rmanager, imanager, pmanager)
            if ret == -1:
                print ('schedule_taskset_main (): create_tasksets () failure')
                return -1
        resource.main_iteration += 1

        taskset_queue = g_iterations[str(resource.main_iteration)]

        index = 0
        prev_taskset = None
        for taskset in taskset_queue:
            if resource.id not in taskset.input.keys():
                print ('schedule_taskset_main (): found main taskset 1', resource.id, resource.main_iteration)
                new_taskset = taskset
                if index == 0:
                    ret = new_taskset.add_input (rmanager, imanager, pmanager, resource)
                    if ret == -1:
                        print ('schedule_taskset_main (): no more images left')
                        return
                else:
                    new_taskset.add_input_taskset (prev_taskset, rmanager, pmanager, resource)
                break
            prev_taskset = taskset
            index += 1
    else:
        taskset_id = str (int (main_taskset_info[1]) + 1)
        taskset_queue = g_iterations[str(main_taskset_iteration)]

        prev_taskset = None
        for taskset in taskset_queue:
            if taskset_id == taskset.tasksetid:
                print ('schedule_taskset_main (): found main taskset 2', resource.id, resource.main_iteration)
                new_taskset = taskset
                new_taskset.add_input_taskset (prev_taskset, rmanager, pmanager, resource)
                break
            prev_taskset = taskset

    new_taskset.submit_main(rmanager, pmanager, [resource_id])
    return 0

def schedule_taskset_support (rmanager, imanager, pmanager, resource_id, resourcetype):
    global g_iterations
    global g_iteration

    resource = rmanager.get_resource(resource_id)

    ##TODO: based on time to completion of current_gpu_taskset, schedule that many images

    current_main_taskset = resource.get_main_taskset(resourcetype)

    print ('schedule_taskset_support ():', resource_id, resourcetype, current_main_taskset)

    support_iteration = 0

    support_iteration = resource.main_iteration + 1

    print ('schedule_taskset_support ():', support_iteration, g_iteration)

    while support_iteration < g_iteration:
        next_taskset_queue = g_iterations[str(support_iteration)]
        prev_taskset = None
        for taskset in next_taskset_queue:
            if resource.id in taskset.input.keys ():
                print ('scheduling_taskset_support ():', 'taskset already complete')
            else:
                if taskset.get_resource_type() != resourcetype:
                    print ('scheduling_taskset_support ():', 'Other stage not executed yet')
                    break
                elif taskset.get_resource_type() == resourcetype:
                    print ('scheduling_taskset_support ():', 'found a support taskset')
                    if prev_taskset != None:
                        taskset.add_input_taskset (prev_taskset, rmanager, pmanager, resource)
                    else:
                        ret = taskset.add_input (rmanager, imanager, pmanager, resource)
                        if ret == -1:
                            print ('schedule_taskset_support (): no more images left')
                            return
                    taskset.submit_support (rmanager, pmanager, [resource_id])
                    return
            print (support_iteration, prev_taskset)
            prev_taskset = taskset
        support_iteration += 1

    #Otherwise create a taskset

    latest_taskset_queue = g_iterations[str(g_iteration - 1)]

    first_taskset = latest_taskset_queue[0]

    if first_taskset.get_resource_type () != resourcetype:
        print ('scheduling_taskset_support ():', 'first taskset type not matching')
        return

    print ('scheduling_taskset_support (): creating new taskset')

    ret = create_tasksets(rmanager, imanager, pmanager)

    if ret == -1:
        print ('schedule_taskset_support (): no more images left')
        return

    latest_taskset_queue = g_iterations[str(g_iteration - 1)]

    first_taskset = latest_taskset_queue[0]

    first_taskset.add_input (rmanager, imanager, pmanager, resource)

    first_taskset.submit_support (rmanager, pmanager, [resource_id])

def status (rmanager):
    global g_iterations
    global g_iteration

    resources = rmanager.get_resources()

    for resource in resources:
        cpu_iteration, current_cpu_taskset = resource.get_current_taskset('CPU')
        gpu_iteration, current_gpu_taskset = resource.get_current_taskset('GPU')

        print ('status ():', resource.id, cpu_iteration, current_cpu_taskset)
        print ('status ():', resource.id, gpu_iteration, current_gpu_taskset)

        if current_cpu_taskset != None:
            cpu_taskset = get_taskset(current_cpu_taskset)
            cpu_taskset.get_status(resource.id)

        if current_gpu_taskset != None:
            gpu_taskset = get_taskset(current_gpu_taskset)
            gpu_taskset.get_status(resource.id)

def is_free (rmanager, imanager, pmanager, resource_id):
    global g_iterations

    print ('is_free ():', resource_id)

    resource = rmanager.get_resource (resource_id)

    cpu_iteration, cpu_current_taskset = resource.get_current_taskset ('CPU')
    gpu_iteration, gpu_current_taskset = resource.get_current_taskset ('GPU')

    print ('is_free ():', cpu_iteration, cpu_current_taskset)
    print ('is_free ():', gpu_iteration, gpu_current_taskset)

    cpu_free = False
    gpu_free = False

    if cpu_iteration == None:
        cpu_free = True

    if gpu_iteration == None:
        gpu_free = True

    if cpu_current_taskset != None:
        cpu_taskset_queue = g_iterations[cpu_current_taskset[0]]

        for taskset in cpu_taskset_queue:
            if taskset.tasksetid == cpu_current_taskset[1]:
                if taskset.input[resource.id]['complete'] == taskset.input[resource.id]['count'] and taskset.input[resource.id]['scheduled'] == 0:
                    cpu_free = True
                    break

    if gpu_current_taskset != None:
        gpu_taskset_queue = g_iterations[gpu_current_taskset[0]]

        for taskset in gpu_taskset_queue:
            if taskset.tasksetid == gpu_current_taskset[1]:
                if taskset.input[resource.id]['complete'] == taskset.input[resource.id]['count'] and taskset.input[resource.id]['scheduled'] == 0:
                    gpu_free = True
                    break

    return cpu_free, gpu_free

rmanager = None
imanager = None
pmanager= None

def get_resource_manager ():
    return rmanager

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

    tasksets[0].submit_main (rmanager, pmanager, [])

    for resource in resources:
        resource.main_iteration = g_iteration - 1

    while True:

        status (rmanager)

        for resource in resources:
            cpu_free, gpu_free = is_free (rmanager, imanager, pmanager, resource.id)

            cpu_iteration, cpu_current_taskset = resource.get_current_taskset ('CPU')
            gpu_iteration, gpu_current_taskset = resource.get_current_taskset ('GPU')

            print (cpu_free, gpu_free)
            print (cpu_current_taskset, gpu_current_taskset)

            if cpu_free and gpu_free:
                if (cpu_iteration != None and cpu_iteration == 'MAIN') or (gpu_iteration != None and gpu_iteration == 'MAIN'):
                    #schedule main taskset and support taskset
                    print ('OAI_scheduler ():', 'scheduling main taskset (with one main)')

                    if cpu_iteration == 'MAIN':
                        ret = schedule_taskset_main (rmanager, imanager, pmanager, resource.id, cpu_current_taskset)
                        if ret == -1:
                            continue

                        cpu_iteration, cpu_current_taskset = resource.get_current_taskset ('CPU')
                        print (cpu_iteration, cpu_current_taskset)
                        if cpu_iteration == None and cpu_current_taskset == None:
                            schedule_taskset_support (rmanager, imanager, pmanager, resource.id, 'CPU')
                    else:
                        ret = schedule_taskset_main (rmanager, imanager, pmanager, resource.id, gpu_current_taskset)
                        if ret == -1:
                            continue

                        gpu_iteration, gpu_current_taskset = resource.get_current_taskset ('GPU')
                        if gpu_iteration == None and gpu_current_taskset == None:
                            schedule_taskset_support (rmanager, imanager, pmanager, resource.id, 'GPU')

                elif (cpu_iteration != None and cpu_iteration == 'SUPPORT') and (gpu_iteration != None and gpu_iteration == 'SUPPORT'):
                    print ('OAI_scheduler ():', 'scheduling main taskset (with both support')

                    cpu_main_taskset = resource.get_main_taskset ('CPU')
                    gpu_main_taskset = resource.get_main_taskset ('GPU')

                    if cpu_main_taskset == None and gpu_main_taskset == None:
                        print ('Ooopsy!!!')

                    if cpu_main_taskset != None:
                        ret = schedule_taskset_main (rmanager, imanager, pmanager, resource.id, cpu_main_taskset)
                        if ret == -1:
                            continue
                        schedule_taskset_support (rmanager, imanager, pmanager, resource.id, 'CPU')
                    else:
                        ret = schedule_taskset_main (rmanager, imanager, pmanager, resource.id, gpu_main_taskset)
                        if ret == -1:
                            continue
                        schedule_taskset_support (rmanager, imanager, pmanager, resource.id, 'GPU')

            elif cpu_free:
                if cpu_iteration == 'MAIN': #TODO:end of the pipeline
                    cpu_taskset = get_taskset (cpu_current_taskset)
                    if cpu_taskset.get_endofpipeline () == True:
                        if gpu_current_taskset[0] == str (resource.main_iteration + 1):
                            resource.main_iteration += 1
                            resource.set_main_taskset (gpu_current_taskset[0], gpu_current_taskset[1], 'GPU')
                            resource.set_current_taskset ('GPU', 'MAIN')
                            #schedule support iteration cpu
                            print ('OAI_scheduler ():', 'scheduling support cpu 1')
                            schedule_taskset_support (rmanager, imanager, pmanager, resource.id, 'CPU')
                        else:
                            print ('OAI_scheduler ():', 'scheduling main 2')
                            ret = schedule_taskset_main (rmanager, imanager, pmanager, resource.id, cpu_current_taskset)
                            if ret == -1:
                                continue
                    else:
                        print ('OAI_scheduler ():', 'scheduling support cpu 2')
                        schedule_taskset_support (rmanager, imanager, pmanager, resource.id, 'CPU')
                elif cpu_iteration == 'SUPPORT':
                    if gpu_iteration != None and gpu_iteration == 'SUPPORT':
                        #schedule main iteration cpu
                        print ('OAI_scheduler ():', 'scheduling main 3')
                        cpu_main_taskset = resource.get_main_taskset ('CPU')
                        gpu_main_taskset = resource.get_main_taskset ('GPU')
                        if cpu_main_taskset != None:
                            ret = schedule_taskset_main (rmanager, imanager, pmanager, resource.id, cpu_main_taskset)
                            if ret == -1:
                                continue
                        else:
                            ret = schedule_taskset_main (rmanager, imanager, pmanager, resource.id, gpu_main_taskset)
                            if ret == -1:
                                continue
                    elif gpu_iteration != None and gpu_iteration == 'MAIN':
                        #schedule support iteration cpu
                        print ('OAI_scheduler ():', 'scheduling support cpu 3')
                        schedule_taskset_support (rmanager, imanager, pmanager, resource.id, 'CPU')
                else:
                    print ('OAI_scheduler ():', 'scheduling support cpu 4')
                    schedule_taskset_support (rmanager, imanager, pmanager, resource.id, 'CPU')
            elif gpu_free:
                if resource.gputype == False:
                    print ('OAI_scheduler ():', 'gpu not available not', resource.id)
                    continue
                if gpu_iteration == 'MAIN':
                    gpu_taskset = get_taskset (gpu_current_taskset)
                    if gpu_taskset.get_endofpipeline () == True:
                        if cpu_current_taskset[0] == str(resource.main_iteration + 1):
                            resource.main_iteration += 1
                            resource.set_main_taskset (cpu_current_taskset[0], cpu_current_taskset[1], 'GPU')
                            resource.set_current_taskset ('CPU', 'MAIN')
                            #schedule support iteration cpu
                            print ('OAI_scheduler ():', 'scheduling support gpu 1')
                            schedule_taskset_support (rmanager, imanager, pmanager, resource.id, 'GPU')
                        else:
                            print ('OAI_scheduler ():', 'scheduling main 4')
                            ret = schedule_taskset_main (rmanager, imanager, pmanager, resource.id, gpu_current_taskset)
                            if ret == -1:
                                continue
                    else:
                        print ('OAI_scheduler ():', 'scheduling support gpu 2')
                        schedule_taskset_support (rmanager, imanager, pmanager, resource.id, 'GPU')
                elif gpu_iteration == 'SUPPORT':
                    if cpu_iteration != None and cpu_iteration == 'SUPPORT':
                        #schedule main iteration gpu
                        print ('OAI_scheduler ():', 'scheduling main 5')
                        cpu_main_taskset = resource.get_main_taskset ('CPU')
                        gpu_main_taskset = resource.get_main_taskset ('GPU')
                        if cpu_main_taskset != None:
                            ret = schedule_taskset_main (rmanager, imanager, pmanager, resource.id, cpu_main_taskset)
                            if ret == -1:
                                continue
                        else:
                            ret = schedule_taskset_main (rmanager, imanager, pmanager, resource.id, gpu_main_taskset)
                            if ret == -1:
                                continue
                    elif cpu_iteration != None and cpu_iteration == 'MAIN':
                        #schedule support iteration gpu
                        print ('OAI_scheduler ():', 'scheduling support gpu 3')
                        schedule_taskset_support (rmanager, imanager, pmanager, resource.id, 'GPU')
                else:
                    print ('OAI_scheduler ():', 'scheduling support gpu 4')
                    schedule_taskset_support (rmanager, imanager, pmanager, resource.id, 'GPU')
        print ('OAI_scheduler ():', 'sleeping for 5')
        time.sleep (5)

    #app1 ('1', get_own_remote_uri()).result()


if __name__ == "__main__":

    configfile = sys.argv[1]

    pipelinefile = sys.argv[2]

    resourcefile = sys.argv[3]

    availablefile = sys.argv[4]

    cost = float (sys.argv[5])

    OAI_scheduler (configfile, pipelinefile, resourcefile, availablefile, cost)
