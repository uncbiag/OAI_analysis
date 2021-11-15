from parslflux.pipeline import PipelineManager

import datetime
from numpy import double
import simpy
from execution_sim import ExecutionSim


class WorkItem:
    def __init__(self, id, data, collectfrom, pipelinestages, resource_id, resourcetype, version, inputlocation):
        self.id = id
        self.version = version
        self.pipelinestages = pipelinestages
        self.data = data
        if collectfrom == None:
            self.collectfrom = resource_id
        else:
            self.collectfrom = collectfrom
        self.status = 'QUEUED'
        self.inputlocation = inputlocation
        self.outputlocation = ''
        self.iscomplete = False
        self.resourceid = resource_id
        self.resourcetype = resourcetype

    def get_id(self):
        return self.id

    def set_resource_id(self, id):
        self.resourceid = id
        if self.collectfrom == None:
            self.collectfrom = id

    def set_complete(self, complete):
        self.iscomplete = complete

    def is_complete(self):
        return self.iscomplete

    def set_status(self, status):
        self.status = status

    def get_status(self):
        return self.status

    def get_lastpipelinestage(self):
        return self.pipelinestages[-1]

    def get_pipelinestages(self):
        return self.pipelinestages

    def update_outputlocation(self, location):
        self.outputlocation = location

    def compose_next_workitem(self, pmanager, resource_id, resourcetype):

        new_pipelinestages = pmanager.get_pipelinestages(self.pipelinestages[-1], resourcetype)

        if new_pipelinestages == None:
            return None

        next_workitem = WorkItem(self.id, self.data, self.resourceid, \
                                 new_pipelinestages, resource_id, resourcetype, \
                                 self.version + 1, self.outputlocation)

        return next_workitem

    def print_data(self):
        # print ('print_data ()', self.id, self.version, self.resourceid)
        pass

    def submit(self, pmanager, timeout, thread_exec, env):
        self.timeout = double(timeout)
        workitem = {}
        workitem['pipelinestages'] = pmanager.encode_pipeline_stages(self.pipelinestages)
        workitem['version'] = str(self.version)
        workitem['collectfrom'] = self.collectfrom
        workitem['workerid'] = self.resourceid

        #print ('interrupting', self.resourceid, self.version, self.resourcetype, thread_exec)
        thread_exec.interrupt (str(self.version))

        # workitem['timeout'] = double (150)
        #self.scheduletime = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        self.scheduletime = env.now
        self.iscomplete = False

        self.status = 'SCHEDULED'

    def cancel(self):
        workitem = {}
        workitem['id'] = self.id
        workitem['version'] = str(self.version)
        workitem['op'] = 'cancel'
        workitem['workerid'] = self.resourceid

        self.status = 'CANCELLING'

    def probe_status(self, thread, outputfile):
        # print ('probe_status ():', self.id, self.version)

        if thread.iscomplete == True:
            str = "probe_status (complete): {} {} {} {} {} success {}".format(self.id, self.version, thread.starttime, thread.endtime, self.resourceid, thread.timeout)
            print (str)
            outputfile.write (str + "\n")
            self.iscomplete = True
            self.starttime = thread.starttime
            self.endtime = thread.endtime
            self.status = 'SUCCESS'
            thread.iscomplete = False
            return True, self.starttime * 3600, self.endtime * 3600, 'SUCCESS', thread.timeout

        return False, None, None, 'INCOMPLETE', 0
