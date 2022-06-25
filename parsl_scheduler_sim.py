import datetime
import time, os, sys

from parslfluxsim.resources_sim import ResourceManager
from parslfluxsim.domain_sim import DomainManager
from oai_scheduler_sim import OAI_Scheduler

from parslflux.pipeline import PipelineManager
from parslfluxsim.input_sim import InputManager2
from plots.plot_prediction_sim import plot_prediction
from performance_analysis import plot_prediction_performance, store_performance_data

import csv
from itertools import zip_longest

from execution_sim import ExecutionSim, ExecutionSimThread
import pandas as pd

import simpy
import matplotlib.pyplot as plt

import numpy as np

class Simulation:
    def __init__(self):
        self.env = simpy.Environment()
        self.r = None

    def setup(self, pipelinefile, configfile, multidomain_resourcefile, \
              max_images, output_file, batchsize):

        self.d = DomainManager(multidomain_resourcefile, self.env)

        self.d.init_resource_model ()

        self.r = ResourceManager(self.env)

        self.i = InputManager2(configfile)

        self.p = PipelineManager(pipelinefile, batchsize, max_images)

        self.p.parse_pipelines (self.r)

        self.scheduler = OAI_Scheduler(self.env)
        self.scheduler.outputfile = output_file

        self.env.process(self.scheduler.run_no_prediction_pin(self.r, self.i, self.p, self.d, exploration=True))

        print ('done')

    def run (self):
        while self.env.peek() < 2000:
            self.env.step()



if __name__ == "__main__":
    #global rmanager, imanager, pmanager

    configfile = "oaiconfig.yml"

    pipelinefile = "parslfluxsim/pipeline2.yml"

    multidomain_resourcefile = "parslfluxsim/md_resources.yml"

    output_directory = "plots/DFS_staging"

    os.makedirs(output_directory, exist_ok=True)

    max_images = [200]

    batchsize = max_images[0]

    no_of_runs = 1

    original_stdout = sys.stdout

    for i in range(len(max_images)):
        for j in range(no_of_runs):
            #sys.stdout = open('output.txt', 'w')
            output_file = open (output_directory+"/"+str(max_images[i])+".txt", "w")
            sim = Simulation ()
            sim.setup (pipelinefile, configfile, multidomain_resourcefile, max_images[i], output_file, batchsize)
            sim.run ()
            #sys.stdout.close ()
            #sys.stdout = original_stdout
            #store_performance_data(algo)

        #plot_prediction_performance ()

    #plot_prediction()
