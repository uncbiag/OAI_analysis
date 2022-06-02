from parslflux.pipeline import PipelineManager

import datetime
from numpy import double
import simpy
from execution_sim import ExecutionSim


class WorkItem:
    def __init__(self, id, data, collectfrom, pipelinestage, resource_id, resourcetype, version, inputlocation):
        self.id = id
        self.version = version
        self.pipelinestage = pipelinestage
        self.phase_index = -1
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
        self.starttime = -1
        self.endtime = -1
        self.priority = int (pipelinestage.priority)
        self.exploration_item = False

    def mark_exploration_item (self, status):
        self.exploration_item = status

    def get_copy (self):
        copy = WorkItem (self.id, self.data, self.collectfrom, self.pipelinestage, self.resourceid, self.resourcetype, self.version, self.inputlocation)
        copy.phase_index = self.phase_index
        copy.exploration_item = self.exploration_item
        return copy

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

    def get_pipelinestage (self):
        return self.pipelinestage.name

    def update_outputlocation(self, location):
        self.outputlocation = location

    def get_next_pipelinestage (self, pmanager, resourcetype):
        next_pipelinestage = pmanager.get_pipelinestage(self.pipelinestage, resourcetype)
        return next_pipelinestage

    def compose_next_workitem(self, resourcetype, next_pipelinestage):

        next_workitem = WorkItem(self.id, self.data, self.resourceid, \
                                 next_pipelinestage, None, resourcetype, \
                                 next_pipelinestage.index, self.outputlocation)
        next_workitem.phase_index = self.phase_index
        next_workitem.starttime = self.starttime
        next_workitem.endtime = self.endtime
        next_workitem.exploration_item = self.exploration_item

        return next_workitem

    def print_data(self):
        pass
        #print ('print_data ()', self.id, self.version, self.phase_index, self.resourceid, self.status, self.pipelinestage.index, self.endtime)

    def submit(self, pmanager, timeout, thread_exec, env):
        self.timeout = double(timeout)
        workitem = {}
        workitem['pipelinestages'] = self.pipelinestage.name
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
            str = "probe_status (complete): {} {} {} {} {} success {}".format(self.id, self.version, thread.starttime,
                                                                              thread.endtime, self.resourceid,
                                                                              thread.timeout)
            print (str)
            outputfile.write (str + "\n")
            self.iscomplete = True
            '''
            self.starttime = thread.starttime
            self.endtime = thread.endtime
            '''
            self.starttime = thread.input_read_starttime
            self.endtime = thread.output_write_endtime
            self.output_write_starttime = thread.output_write_starttime
            self.output_write_endtime = thread.output_write_endtime
            self.input_read_starttime = thread.input_read_starttime
            self.input_read_endtime = thread.input_read_endtime

            self.status = 'SUCCESS'
            thread.iscomplete = False
            print ('probe_status ()', self.id, self.starttime, self.endtime,
                   self.input_read_starttime, self.input_read_endtime, self.output_write_starttime, self.output_write_endtime)
            return True, self.starttime, self.endtime, 'SUCCESS', thread.timeout

        return False, None, None, 'INCOMPLETE', 0
