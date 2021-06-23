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

from resources import ResourceManager
from pipeline import PipelineManager
from taskset import Taskset
from input import InputManager

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
                    scheduler_options=options,
                    launcher=SrunLauncher(),
                ),
            )
        ],
        app_cache=False,
    )
    return inspector_config


@bash_app
def app1 (pipelineStage, own_uri):
    return '~/local/bin/flux start python3.6 flux_wrapper.py {} {}'.format(pipelineStage, own_uri)

def launch_workers (resources):
    for resource in resources:
        parsl.clear()
        options = '#SBATCH -w ' + resource.hostname
        config = get_launch_config (options)
        parsl.load (config)
        app1 (resource.hostname, get_own_remote_uri())

def setup (resourcefile, pipelinefile, inputfile, availablefile):
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

    i = InputManager (inputfile)
    i.parse_input ()

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
            if len(g_tasksets) > 0:
                prev_ts = g_tasksets[-1]
                Ti.add_input_taskset(prev_ts, rmanager, pmanager)
            else:
                Ti.add_input(rmanager, imanager, pmanager)

            g_tasksets.append(Ti)

            Ti.print_data()

        if endofpipeline == True:
            g_iterations[str(g_iteration)] = g_tasksets
            g_iteration += 1
            g_tasksets = []
            break

def delete_taskset_queue ():
    global g_iterations
    global g_iteration

    del g_iterations[g_iteration]
    g_iteration -= 1

def free_resources (rmanager, imanager, pmanager):
    global g_iterations

    free_cpus = []
    free_gpus = []

    for resource in rmanager.get_resources ():

        current_cpu_main_taskset = resource.get_main_taskset('CPU')
        current_gpu_main_taskset = resource.get_main_taskset('GPU')

        current_cpu_support_taskset = resource.get_support_taskset('CPU')
        current_gpu_support_taskset = resource.get_support_taskset('GPU')

        no_main_taskset = False
        no_support_taskset = False

        if current_cpu_main_taskset == None:
            no_main_taskset = True
        else:
            cpu_taskset_queue = g_iterations[current_cpu_support_taskset[0]]
            for taskset in cpu_taskset_queue:
                if taskset.id == current_cpu_support_taskset[1]:
                    if taskset.input[resource.id]['complete'] == taskset.input[resource.id]['count'] and taskset.input[resource.id]['scheduled'] == 0:
                        no_main_taskset = True
                        break

        if current_cpu_support_taskset == None:
            no_support_taskset = True
        else:
            cpu_taskset_queue = g_iterations[current_cpu_main_taskset[0]]
            for taskset in cpu_taskset_queue:
                if taskset.id == current_cpu_main_taskset[1]:
                    if taskset.input[resource.id]['complete'] == taskset.input[resource.id]['count'] and taskset.input[resource.id]['scheduled'] == 0:
                        no_support_taskset = True
                        break

        if no_main_taskset and no_support_taskset:
            free_cpus.append(resource.id)

        no_main_taskset = False
        no_support_taskset = False

        if current_gpu_main_taskset == None:
            no_main_taskset = True
        else:
            gpu_taskset_queue = g_iterations[current_gpu_support_taskset[0]]
            for taskset in gpu_taskset_queue:
                if taskset.id == current_gpu_support_taskset[1]:
                    if taskset.input[resource.id]['complete'] == taskset.input[resource.id]['count'] and taskset.input[resource.id]['scheduled'] == 0:
                        no_main_taskset = True
                        break

        if current_gpu_support_taskset == None:
            no_support_taskset = True
        else:
            gpu_taskset_queue = g_iterations[current_gpu_main_taskset[0]]
            for taskset in gpu_taskset_queue:
                if taskset.id == current_gpu_main_taskset[1]:
                    if taskset.input[resource.id]['complete'] == taskset.input[resource.id]['count'] and taskset.input[resource.id]['scheduled'] == 0:
                        no_support_taskset = True
                        break

        if no_main_taskset and no_support_taskset:
            free_gpus.append(resource.id)

    return free_cpus, free_gpus

def get_taskset (taskset_info):
    global g_iterations

    iteration = taskset_info[0]
    taskset_id = taskset_info[1]

    taskset_queue = g_iterations[str(iteration)]

    for taskset in taskset_queue:
        if taskset_id == taskset.id:
            return taskset

    return None

def schedule_taskset_main (rmanager, pmanager, resource_id, current_taskset):
    global g_iterations
    global g_iteration

    resource = rmanager.get_resource(resource_id)

    current_taskset_iteration = current_taskset[0]

    new_taskset = None

    if current_taskset_iteration != resource.main_iteration:
        taskset_queue = g_iterations[str(resource.main_iteration)]
        for taskset in taskset_queue:
            if taskset.input[resource.id]['complete'] < \
                    taskset.input[resource.id]['count']:
                new_taskset = taskset
                break
    else:
        taskset_id = current_taskset[1] + 1
        taskset_queue = g_iterations[str(current_taskset_iteration)]

        for taskset in taskset_queue:
            if taskset_id == taskset.id:
                new_taskset = taskset
                break

    new_taskset.submit_main(rmanager, pmanager, [resource_id])

def schedule_taskset_support (rmanager, imanager, pmanager, resource_id, resourcetype):
    global g_iterations
    global g_iteration

    resource = rmanager.get_resource(resource_id)

    ##TODO: based on time to completion of current_gpu_taskset, schedule that many images

    current_main_taskset = resource.get_main_taskset(resourcetype)
    current_support_taskset = resource.get_support_taskset(resourcetype)

    support_iteration = 0

    if current_support_taskset != None:
        support_iteration = current_support_taskset[0] + 1
    else:
        support_iteration = resource.main_iteration + 1

    while support_iteration < g_iteration - 1:
        next_taskset_queue = g_iterations[str(support_iteration)]
        for taskset in next_taskset_queue:
            if taskset.input[resource.id]['complete'] == taskset.input[resource.id]['count']:
                continue
            if taskset.get_resource_type() != resourcetype and taskset.input[resource.id]['complete'] < taskset.input[resource.id]['count']:
                print ('Other stage not executed yet')
                taskset.print_data()
                break
            elif taskset.get_resource_type() == resourcetype:
                print ('found a support taskset')
                taskset.print_data()
                taskset.submit_support(rmanager, pmanager, [resource_id])
                return
        support_iteration += 1

    #Otherwise create a taskset
    create_tasksets(rmanager, imanager, pmanager)

    latest_taskset_queue = g_iterations[str(g_iteration - 1)]

    first_taskset = latest_taskset_queue[0]

    if first_taskset.get_resource_type () != resourcetype:
        delete_taskset_queue()
        return

    first_taskset.submit ([resource_id])

def status (rmanager):
    global g_iterations
    global g_iteration

    resources = rmanager.get_resource()

    for resource in resources:
        current_cpu_taskset = resource.get_current_taskset('CPU')
        current_gpu_taskset = resource.get_current_taskset('GPU')

        if current_cpu_taskset != None:
            cpu_taskset = get_taskset(current_cpu_taskset)
            cpu_taskset.get_status()

        if current_gpu_taskset != None:
            gpu_taskset = get_taskset(current_gpu_taskset)
            gpu_taskset.get_status()

def OAI_scheduler (inputfile, pipelinefile, resourcefile, availablefile, cost):

    global g_iterations
    global g_iteration

    rmanager, imanager, pmanager = setup (resourcefile, pipelinefile, inputfile, availablefile)

    print ('sleeping for 10 secs')

    time.sleep (10)

    create_tasksets(rmanager, imanager, pmanager)

    tasksets = g_iterations['0']

    #submit first cpu & gpu pipeline stages.

    tasksets[0].submit_main (rmanager, pmanager, [])

    print ('taskset submitted')

    return

    resources = rmanager.get_resources()

    for resource in resources:
        resource.main_iteration = g_iteration - 1

    while True:

        status ()

        free_cpus, free_gpus = free_resources ()

        if len(free_cpus) > 0:
            for free_cpu in free_cpus:
                current_cpu_taskset = free_cpu.get_current_taskset('CPU')
                if current_cpu_taskset[0] == free_cpu.main_iteration:
                    cpu_taskset = get_taskset (current_cpu_taskset)
                    if cpu_taskset.get_endofpipeline() == False:
                        if free_cpu.id in free_gpus:
                            schedule_taskset_main (rmanager, pmanager, free_cpu.id, current_cpu_taskset) #schedule next GPU taskset from main iteration
                            #remove gpu.id from free_gpus
                            free_gpus.remove(free_cpu.id)

                        schedule_taskset_support (rmanager, free_cpu.id, 'CPU') #schedule support cpu taskset

                        free_cpus.remove(free_cpu.id)
                    else:
                        if free_cpu.main_iteration == g_iteration - 1:
                            create_tasksets(rmanager, imanager, pmanager)  # create tasksets if we don't have anymore iterations left
                        # change main_iteration to next taskset queue
                        if g_iterations[str(free_cpu.main_iteration)][0].get_resource_type() == 'GPU':
                            if free_cpu.id in free_gpus:
                                free_cpu.main_iteration += 1
                                schedule_taskset_main (rmanager, pmanager, free_cpu.id, current_cpu_taskset)
                                free_gpus.remove(free_cpu.id)

                                schedule_taskset_support(rmanager, free_cpu.id, 'CPU')  # schedule support cpu taskset
                        else:
                            free_cpu.main_iteration += 1
                            schedule_taskset_main(rmanager, pmanager, free_cpu.id, current_cpu_taskset)

                        free_cpus.remove(free_cpu.id)
                else:
                    if free_cpu.id not in free_gpus:
                        schedule_taskset_support(rmanager, free_cpu.id, 'CPU') # continue scheduling support cpu taskset if gpu phase still executing
                        free_cpus.remove(free_cpu.id)

        if len(free_gpus) > 0:
            for free_gpu in free_gpus:
                current_gpu_taskset = free_gpu.get_current_taskset('GPU')
                if current_gpu_taskset[0] == free_gpu.main_iteration:
                    gpu_taskset = get_taskset (current_gpu_taskset)
                    if gpu_taskset.get_endofpipeline() == False:
                        if free_gpu.id in free_cpus:
                            schedule_taskset_main (rmanager, pmanager, free_cpu.id, current_gpu_taskset)

                        schedule_taskset_support (rmanager, free_cpu.id, 'GPU')
                    else:
                        # change main_iteration to next taskset queue
                        if free_gpu.main_iteration == g_iteration - 1:
                            create_tasksets(rmanager, imanager, pmanager)

                        if g_iterations[str(free_gpu.main_iteration)][0].get_resource_type() == 'CPU':
                            if free_gpu.id in free_cpus:
                                if __name__ == '__main__':
                                    free_gpu.main_iteration += 1
                                    schedule_taskset_main(rmanager, pmanager, free_cpu.id, current_gpu_taskset)

                                    schedule_taskset_support(rmanager, free_cpu.id, 'GPU')
                        else:
                            free_gpu.main_iteration += 1
                            schedule_taskset_main(rmanager, pmanager, free_cpu.id, current_gpu_taskset)
                else:
                    if free_gpu.id not in free_cpus:
                        schedule_taskset_support(rmanager, free_gpu.id, 'GPU')

    #app1 ('1', get_own_remote_uri()).result()


if __name__ == "__main__":
    inputfile = sys.argv[1]

    pipelinefile = sys.argv[2]

    resourcefile = sys.argv[3]

    availablefile = sys.argv[4]

    cost = float (sys.argv[5])

    OAI_scheduler (inputfile, pipelinefile, resourcefile, availablefile, cost)
