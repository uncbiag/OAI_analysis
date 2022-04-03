import datetime
import time, os, sys

from parslfluxsim.resources_sim import ResourceManager
from oai_scheduler_sim import OAI_Scheduler
from parslflux.pipeline import PipelineManager
from parslfluxsim.input_sim import InputManager2
from parslfluxsim.performance_sim import read_performance_data
from plots.plot_prediction_sim import plot_prediction
from performance import plot_prediction_performance, store_performance_data

import csv
from itertools import zip_longest

from execution_sim import ExecutionSim, ExecutionSimThread
import pandas as pd

import simpy
import matplotlib.pyplot as plt

import numpy as np

'''
def read_performance_data (rmanager, performancefile):
    completion_file = open(performancefile, "r")
    completion_lines = completion_file.readlines()

    min_submit_time = None

    images = []

    for completion_line in completion_lines:
        _, _, imageid, version, submitdate, submithour, startdate, starthour, enddate, endhour, resourceid, _ = completion_line.split(
            ' ',
            11)

        resourceid = resourceid.strip()

        submittime_s = submitdate + ' ' + submithour
        starttime_s = startdate + ' ' + starthour
        endtime_s = enddate + ' ' + endhour

        submittime = datetime.datetime.strptime(submittime_s,
                                                '%Y-%m-%d %H:%M:%S').timestamp()
        starttime = datetime.datetime.strptime(starttime_s,
                                               '%Y-%m-%d %H:%M:%S').timestamp()
        endtime = datetime.datetime.strptime(endtime_s, '%Y-%m-%d %H:%M:%S').timestamp()

        if min_submit_time == None:
            min_submit_time = submittime
        elif min_submit_time > submittime:
            min_submit_time = submittime

        images.append([imageid, version, resourceid, submittime, starttime, endtime])

    for image in images:
        image[3] -= min_submit_time
        image[4] -= min_submit_time
        image[5] -= min_submit_time

    resource_performance = {}

    for image in images:
        resource_id = image[2]
        version = image[1]

        resource = rmanager.get_resource_exist(resource_id)

        if resource == None:
            print ('resource id ', resource_id, 'invalid')

        if int (version)% 2 == 0:
            resourcetype = resource.cpu.name
        else:
            resourcetype = resource.gpu.name

        if resourcetype not in resource_performance:
            resource_performance[resourcetype] = {}
            if version not in resource_performance[resourcetype]:
                resource_performance[resourcetype][version] = []
                resource_performance[resourcetype][version].append(image[5] - image[3])
            else:
                resource_performance[resourcetype][version].append(image[5] - image[3])
        else:
            if version not in resource_performance[resourcetype]:
                resource_performance[resourcetype][version] = []
                resource_performance[resourcetype][version].append(image[5] - image[3])
            else:
                resource_performance[resourcetype][version].append(image[5] - image[3])

    for resourceid in resource_performance.keys():
        version_data = resource_performance[resourceid]['1']
        resource_performance[resourceid]['1'] = resource_performance[resourceid]['4']
        resource_performance[resourceid]['4'] = version_data

    #print (resource_performance)

    from scipy.stats import anderson

    for resourcetype in resource_performance.keys():
        csv_columns = list (resource_performance[resourcetype].keys())

        for version in resource_performance[resourcetype].keys():
            csv_filename = resourcetype + version + '.csv'
            data = {version:resource_performance[resourcetype][version]}
            dataframe = pd.DataFrame.from_dict(data)
            print (dataframe)
            dataframe.to_csv(csv_filename, index=False)

            #print (resourcetype, version, anderson(resource_performance[resourcetype][version]))
            #plt.hist(resource_performance[resourcetype][version], bins=30)
            #plt.show()

    return resource_performance
'''

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


class Simulation:
    def __init__(self):
        self.env = simpy.Environment()
        self.r = None

    def setup(self, resourcefile, pipelinefile, configfile, availablefile, \
              max_images, output_file, prediction, batchsize, no_of_prediction_phases, algo, reconfiguration_time_delta):

        self.r = ResourceManager(resourcefile, availablefile, self.env)

        print (self.r)

        self.r.parse_resources()

        #self.r.purge_resources()

        self.performancedata = read_performance_data()

        print (self.performancedata)

        self.i = InputManager2(configfile)

        self.p = PipelineManager(pipelinefile, cost, batchsize, max_images)

        self.p.parse_pipelines(self.r)

        self.p.build_phases()

        self.scheduler = OAI_Scheduler(self.env)

        self.worker_threads = {}

        for resource in self.r.get_resources():
            if resource.cpu != None:
                cpu_thread = ExecutionSimThread(self.env, resource, 'CPU', self.performancedata)
            else:
                cpu_thread = None
            if resource.gpu != None:
                gpu_thread = ExecutionSimThread(self.env, resource, 'GPU', self.performancedata)
            else:
                gpu_thread = None
            self.worker_threads[resource.id] = [cpu_thread, gpu_thread]

        self.workers = {}

        for resource_id in self.worker_threads.keys():
            cpu_thread = self.worker_threads[resource_id][0]
            gpu_thread = self.worker_threads[resource_id][1]

            if cpu_thread != None:
                cpu_thread_exec = ExecutionSim(self.env, cpu_thread)
            else:
                cpu_thread_exec = None
            if gpu_thread != None:
                gpu_thread_exec = ExecutionSim(self.env, gpu_thread)
            else:
                gpu_thread_exec = None
            self.workers[resource_id] = [cpu_thread_exec, gpu_thread_exec]

        self.scheduler.workers = self.workers
        self.scheduler.worker_threads = self.worker_threads
        self.scheduler.outputfile = output_file
        self.scheduler.performancedata = self.performancedata
        self.scheduler.no_of_prediction_phases = no_of_prediction_phases
        self.scheduler.batchsize = batchsize
        self.scheduler.algo = algo
        self.scheduler.reconfiguration_time_delta = reconfiguration_time_delta
        if prediction == True:
            self.env.process(self.scheduler.run_prediction(self.r, self.i, self.p))
        else:
            self.env.process(self.scheduler.run_no_prediction(self.r, self.i, self.p))

        print ('done')

    def run (self):
        while self.env.peek() < 2000:
            self.env.step()



if __name__ == "__main__":
    #global rmanager, imanager, pmanager

    configfile = "oaiconfig.yml"

    pipelinefile = "parslflux/pipeline.yml"

    resourcefile = "parslflux/resources2.yml"

    availablefile = "parslflux/available.yml"

    cost = 1000

    output_directory = "plots/DFS_staging"

    no_of_prediction_phases = 1

    os.makedirs(output_directory, exist_ok=True)

    #max_images = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    max_images = [200]

    batchsize = max_images[0]

    prediction = False

    if prediction == True:
        batchsize = 20
    else:
        batchsize = max_images[0]

    #reconfiguration_algos = ['base', 'down', 'overallocation', 'underallocation']
    reconfiguration_algos = ['overallocation']
    no_of_runs = 50

    reconfiguration_time_delta = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    original_stdout = sys.stdout
    for i in range(len(max_images)):
        for reconfiguration_time in reconfiguration_time_delta:
            for algo in reconfiguration_algos:
                for j in range(no_of_runs):
                    sys.stdout = open('output.txt', 'w')
                    output_file = open (output_directory+"/"+str(max_images[i])+".txt", "w")
                    sim = Simulation ()
                    sim.setup(resourcefile, pipelinefile, configfile, availablefile, max_images[i], output_file, prediction, batchsize, no_of_prediction_phases, algo, reconfiguration_time)
                    sim.run ()
                    sys.stdout.close()
                    sys.stdout = original_stdout
                    print(algo, reconfiguration_time, 'simulation ', j, 'complete')
                print ('algorithm', algo, 'complete')
                store_performance_data(algo)
            print ('reconfiguration_time', reconfiguration_time, 'complete')

        #plot_prediction_performance ()

    #plot_prediction()
