import simpy
import random

class ExecutionSim:
    def __init__ (self, env, app):
        self.env = env
        self.app = app
        self.exe = self.env.process (self.app.run ())
    def get_exec (self):
        return self.exe

class ExecutionSimThread:
    def __init__ (self, env, resourceid, resourcetype, performancedata):
        self.env = env
        self.resourcetype = resourcetype
        self.resourceid = resourceid
        self.performancedata = performancedata
        self.iscomplete = False
        self.interrupts = 0

    def run (self):
        print (self.resourceid, self.resourcetype, 'started')
        while True:
            try:
                #print ('sleeping')
                yield self.env.timeout (0.25)
                continue
            except simpy.Interrupt as interrupt:
                self.interrupts += 1
                resource_id, version = interrupt.cause.split(":")
                #print (self.resourceid, self.resourcetype, version, self.interrupts)
                startime = self.env.now
                timeouts = self.performancedata[resource_id][version]
                #timeout_max = max(timeouts)
                #timeout_min = min(timeouts)
                #timeout = random.randrange(timeout_min, timeout_max, 1)
                timeout = sum(timeouts)/len(timeouts)
                #print(self.resourceid, version, timeout_min, timeout_max, timeout/3600, self.interrupts)
                yield self.env.timeout (timeout/3600)
                endtime = self.env.now
                #print (resource_id, version, startime, endtime, 'complete', self.interrupts)
                self.iscomplete = True
                self.timeout = timeout
                self.starttime = startime
                self.endtime = endtime


