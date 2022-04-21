import simpy
import random
from scipy.stats import *
import statistics
import numpy as np
class ExecutionSim:
    def __init__ (self, env, app):
        self.env = env
        self.app = app
        self.exe = self.env.process (self.app.run ())
    def get_exec (self):
        return self.exe

class ExecutionSimThread:
    def __init__ (self, env, resource, resourcetype, performancedata, provision_type, provision_time):
        self.env = env
        self.resourcetype = resourcetype
        self.provision_type = provision_type
        self.provision_time = provision_time
        self.resource = resource
        if self.resourcetype == 'CPU':
            self.resourcename = self.resource.cpu.name
        else:
            self.resourcename = self.resource.gpu.name
        self.performancedata = performancedata
        self.iscomplete = False
        self.interrupts = 0
        self.requesttime = self.env.now

        self.distributions = {}

        for resourcename in performancedata.keys ():
            if self.resourcename == resourcename:
                for version in performancedata[resourcename].keys ():
                    dist, shape, location, scale = performancedata[resourcename][version][0], performancedata[resourcename][version][1], \
                        performancedata[resourcename][version][2], performancedata[resourcename][version][3]
                    self.distributions[version] = {}
                    '''
                    if dist == 'lognorm':
                        self.distributions[version]['0'] = lognorm.rvs (shape, location, scale, 10000)
                    elif dist == 'gamma':
                        self.distributions[version]['0'] = gamma.rvs (shape, location, scale, 10000)
                    elif dist == 'gennorm':
                        self.distributions[version]['0'] = gennorm.rvs (shape, location, scale, 10000)
                    '''
                    self.distributions[version]['1'] = [dist, shape, location, scale]
                    #print(self.resourcename, version, min(self.distributions[version]['0']), max(self.distributions[version]['0']), statistics.mean(self.distributions[version]['0']), np.median(self.distributions[version]['0']), mode(self.distributions[version]['0']))


    def get_timeout (self, version):
        dist = self.distributions[version]['1'][0]
        shape = self.distributions[version]['1'][1]
        location = self.distributions[version]['1'][2]
        scale = self.distributions[version]['1'][3]

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
        print (self.resource.id, self.resourcetype, 'started', self.env.now)
        while True:
            try:
                #print ('sleeping')
                yield self.env.timeout (0.25)
                continue
            except simpy.Interrupt as interrupt:
                if interrupt.cause == 'cancel':
                    print (self.resource.id, self.resourcetype, 'exiting...')
                    break
                else:
                    self.interrupts += 1
                    version = interrupt.cause
                    #print (self.resourceid, self.resourcetype, version, self.interrupts)
                    startime = self.env.now
                    timeout = self.get_timeout(version)
                    #timeout = self.distributions[version][random.randrange(len(self.distributions))]
                    #timeouts = self.performancedata[self.resourcename][version]
                    #timeout_max = max(timeouts)
                    #timeout_min = min(timeouts)
                    #timeout = random.randrange(timeout_min, timeout_max, 1)
                    #timeout = sum(timeouts)/len(timeouts)
                    #print(self.resourceid, version, timeout_min, timeout_max, timeout/3600, self.interrupts)
                    yield self.env.timeout (timeout/3600)
                    endtime = self.env.now
                    #print (resource_id, version, startime, endtime, 'complete', self.interrupts)
                    self.iscomplete = True
                    self.timeout = timeout
                    self.starttime = startime
                    self.endtime = endtime


