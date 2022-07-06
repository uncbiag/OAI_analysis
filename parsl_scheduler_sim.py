import os, sys

from parslfluxsim.resources_sim import ResourceManager
from parslfluxsim.domain_sim import DomainManager
from oai_scheduler_sim import OAI_Scheduler

from parslfluxsim.pipeline_sim import PipelineManager
from parslfluxsim.input_sim import InputManager2
from parslfluxsim.allocator_sim import Allocator
from parslfluxsim.scaling_sim import Scaler

import simpy


class Simulation:
    def __init__(self):
        self.env = simpy.Environment()

    def setup(self, pipelinefile, configfile, multidomain_resourcefile, \
              max_images, output_file, batchsize, interval):

        self.d = DomainManager(multidomain_resourcefile, self.env)

        self.d.init_resource_model ()

        self.r = ResourceManager(self.env)

        #self.i = InputManager2(configfile)
        self.i = InputManager2(batchsize)

        self.p = PipelineManager(pipelinefile, batchsize, max_images, self.env)

        self.p.parse_pipelines (self.r)

        self.scheduler = OAI_Scheduler(self.env)
        self.scheduler.outputfile = output_file

        self.a = Allocator (self.env)

        self.s = Scaler (self.env, interval)

        return self.r, self.i, self.p, self.d, self.a, self.s

    def run (self):
        self.env.run()
        '''
        while self.env.peek() < 2000:
            self.env.step()
            print ('run()', self.env.peek())
        '''



if __name__ == "__main__":
    #global rmanager, imanager, pmanager

    configfile = "oaiconfig.yml"

    pipelinefile = "parslfluxsim/pipeline2.yml"

    multidomain_resourcefile = "parslfluxsim/md_resources.yml"

    output_directory = "plots/DFS_staging"

    deadline = 3
    interval = 0.5

    os.makedirs(output_directory, exist_ok=True)

    max_images = [2000]

    no_of_runs = 1

    original_stdout = sys.stdout

    for i in range(len(max_images)):
        batchsize = max_images[i]
        for j in range(no_of_runs):
            sys.stdout = open('output.txt', 'w')
            output_file = open (output_directory+"/"+str(max_images[i])+".txt", "w")
            sim = Simulation ()
            r,i,p,d,a,s = sim.setup (pipelinefile, configfile, multidomain_resourcefile, max_images[i], output_file, batchsize,\
                                     interval)
            sim.env.process(sim.scheduler.run_no_prediction_pin_core(r, i, p, d, a, s, exploration=True))
            sim.run ()
            print('actual execution begins', sim.env.now)
            r.reset()
            i.reset()
            p.reset()
            d.reset()
            s.reset()
            print ('actual execution begins', sim.env.now)
            s.set_deadline(deadline)
            sim.scheduler.reset ()
            sim.env.process(sim.scheduler.run_no_prediction_pin_core(r, i, p, d, a, s, exploration=False))
            sim.run()

            sys.stdout.close ()
            sys.stdout = original_stdout
            #store_performance_data(algo)

        #plot_prediction_performance ()

    #plot_prediction()
