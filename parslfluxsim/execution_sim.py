import simpy
import random
from scipy.stats import *
import statistics
import numpy as np

class ExecutionSim:
    def __init__ (self, env, app, reservation_timeout):
        self.env = env
        self.app = app
        self.status = 'RUNNING'
        self.exe = self.env.process (self.app.run ())
        if reservation_timeout != -1:
            self.reservation = self.env.process (self.reservation_timeout (reservation_timeout, self.exe))

    def reservation_timeout (self, timeout, exec):
        try:
            yield self.env.timeout(timeout)
        except simpy.Interrupt as interrupt:
            print ('reservation_timeout ()', timeout)

        if exec.is_alive == True:
            exec.interrupt ('timeout')
            self.status = 'CANCELLED'
        else:
            print ('reservation_timeout ()', 'run already dead')
        print ('reservation_timeout ()', 'exiting', self.env.now)

    def get_exec (self):
        return self.exe

class ExecutionSimThread:
    def __init__ (self, env, resource, resourcetype, performancedist, provision_type, provision_time, domain_id):
        self.env = env
        self.resourcetype = resourcetype
        self.provision_type = provision_type
        self.provision_time = provision_time
        self.resource = resource
        if self.resourcetype == 'CPU':
            self.resourcename = self.resource.cpu.name
        else:
            self.resourcename = self.resource.gpu.name
        self.performancedist = performancedist
        self.iscomplete = False
        self.interrupts = 0
        self.requesttime = self.env.now
        self.domain_id = domain_id

        self.distributions = {}


        for pipelinestage in performancedist.keys ():
            dist, shape, location, scale = performancedist[pipelinestage][0], performancedist[pipelinestage][1], \
                                            performancedist[pipelinestage][2], performancedist[pipelinestage][3]
            self.distributions[pipelinestage] = {}
            '''
            if dist == 'lognorm':
                self.distributions[version]['0'] = lognorm.rvs (shape, location, scale, 10000)
            elif dist == 'gamma':
                self.distributions[version]['0'] = gamma.rvs (shape, location, scale, 10000)
            elif dist == 'gennorm':
                self.distributions[version]['0'] = gennorm.rvs (shape, location, scale, 10000)
            '''
            self.distributions[pipelinestage]['1'] = [dist, shape, location, scale]
            #print(self.resourcename, version, min(self.distributions[version]['0']), max(self.distributions[version]['0']), statistics.mean(self.distributions[version]['0']), np.median(self.distributions[version]['0']), mode(self.distributions[version]['0']))


    def get_timeout (self, pipelinestage):
        dist = self.distributions[pipelinestage]['1'][0]
        shape = self.distributions[pipelinestage]['1'][1]
        location = self.distributions[pipelinestage]['1'][2]
        scale = self.distributions[pipelinestage]['1'][3]

        if dist == 'gamma':
            return gamma.rvs(shape, location, scale, 1)[0]
        elif dist == 'lognorm':
            return lognorm.rvs(shape, location, scale, 1)[0]
        elif dist == 'gennorm':
            return gennorm.rvs(shape, location, scale, 1)[0]

    def run (self):
        try:
            yield self.env.timeout(self.provision_time)
        except simpy.Interrupt as interrupt:
            print ('unexpected: cancelled before provision')
            self.resource.print_data()
            return
        self.startup_time = self.env.now
        self.resource.set_active (True)
        self.resource.set_idle_start_time(self.resourcetype, self.env.now)
        print (self.domain_id, self.resource.id, self.resourcetype, 'started', self.env.now)

        while True:
            try:
                #print ('sleeping')
                yield self.env.timeout (0.25)
                continue
            except simpy.Interrupt as interrupt:
                if interrupt.cause == 'timeout':
                    print (self.domain_id, self.resource.id, self.resourcetype, 'timeout', 'exiting...')
                    break
                elif interrupt.cause == 'cancel':
                    print (self.domain_id, self.resource.id, self.resourcetype, 'cancelled', 'exiting...')
                    break
                else:
                    self.interrupts += 1
                    info = interrupt.cause.split (':')
                    pipelinestage = info[0]
                    input_read_time = float (info[1])
                    output_write_time = float (info[2])

                    #print (self.resourceid, self.resourcetype, version, self.interrupts)
                    startime = self.env.now

                    timeout = self.get_timeout(pipelinestage)

                    input_read_starttime = self.env.now

                    exec_starttime = input_read_starttime + input_read_time
                    exec_endtime = exec_starttime + (timeout/3600)

                    output_write_endtime = exec_endtime + output_write_time

                    #print (version, 'sleeping', timeout/3600)
                    try:
                        yield self.env.timeout ((timeout + input_read_time + output_write_time)/3600)
                    except simpy.Interrupt as interrupt:
                        if interrupt.cause == 'cancel':
                            print(self.domain_id, self.resource.id, self.resourcetype, 'cancelled', 'exiting...')
                            break
                        elif interrupt.cause == 'timeout':
                            print(self.domain_id, self.resource.id, self.resourcetype, 'timeout', 'exiting...')
                            break
                    endtime = self.env.now
                    #print (resource_id, version, startime, endtime, 'complete', self.interrupts)
                    self.iscomplete = True
                    self.timeout = timeout
                    self.starttime = startime
                    self.endtime = endtime

                    self.exec_starttime = exec_starttime
                    self.exec_endtime = exec_endtime
                    self.input_read_starttime = input_read_starttime
                    self.input_read_endtime = exec_starttime
                    self.output_write_starttime = exec_endtime
                    self.output_write_endtime = output_write_endtime