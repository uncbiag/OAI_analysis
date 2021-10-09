from parslflux.pipeline import PipelineManager
import datetime
import flux
from numpy import double

class WorkItem:
    def __init__ (self, id, data, collectfrom, pipelinestages, resource_id, resourcetype, version, inputlocation):
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

    def get_id (self):
        return self.id

    def set_resource_id (self, id):
        self.resourceid = id
        if self.collectfrom == None:
            self.collectfrom = id

    def set_complete (self, complete):
        self.iscomplete = complete

    def is_complete (self):
        return self.iscomplete

    def set_status (self, status):
        self.status = status

    def get_status (self):
        return self.status

    def get_lastpipelinestage (self):
        return self.pipelinestages[-1]

    def get_pipelinestages (self):
        return self.pipelinestages

    def update_outputlocation (seld, location):
        self.outputlocation = location

    def compose_next_workitem (self, pmanager, resource_id, resourcetype):

        new_pipelinestages = pmanager.get_pipelinestages (self.pipelinestages[-1], resourcetype)

        if new_pipelinestages == None:
            return None
        
        next_workitem = WorkItem (self.id, self.data, self.resourceid, \
                                  new_pipelinestages, resource_id, resourcetype, \
                                  self.version + 1, self.outputlocation)

        return next_workitem

    def print_data (self):
        #print ('print_data ()', self.id, self.version, self.resourceid)
        pass

    def submit (self, pmanager, timeout):
        self.timeout = double (timeout)
        workitem = {}
        workitem['pipelinestages'] = pmanager.encode_pipeline_stages(self.pipelinestages)
        workitem['id'] = self.id
        workitem['version'] = str(self.version)
        workitem['data'] = self.data
        workitem['resourcetype'] = self.resourcetype
        workitem['inputlocation'] = self.inputlocation
        workitem['collectfrom'] = self.collectfrom
        workitem['workerid'] = self.resourceid
        workitem['op'] = 'add'
        workitem['timeout'] = double (timeout)
        #workitem['timeout'] = double (150)
        self.scheduletime = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        self.scheduletime = datetime.datetime.strptime (self.scheduletime, '%Y-%m-%d %H:%M:%S')
        self.iscomplete = False

        print ('submit (): ', workitem)

        f = flux.Flux ()
        f.rpc (b"parslmanager.workitem.submit", workitem)

        self.status = 'SCHEDULED'

    def cancel (self):
        workitem = {}
        workitem['id'] = self.id
        workitem['version'] = str (self.version)
        workitem['op'] = 'cancel'
        workitem['workerid'] = self.resourceid

        print ('cancel ():', workitem)

        f = flux.Flux ()
        f.rpc (b"parslmanager.workitem.submit", workitem)

        self.status = 'CANCELLING'

    def probe_status (self):
        #print ('probe_status ():', self.id, self.version)
        f = flux.Flux ()
        r = f.rpc ("parslmanager.workitem.status", {"workerid":self.resourceid, \
                   "id":self.id, "version":str(self.version)}).get ()

        #print ('report', r)

        report = r['report']

        if type (report) is not dict and report == 'empty':
            #print ('empty report')
            return False, None, None, 'INCOMPLETE', 0

        status = report['status']

        self.iscomplete = True

        if status == 'SUCCESS':
            self.starttime = datetime.datetime.strptime (report['starttime'], '%Y-%m-%d %H:%M:%S')
            self.endtime = datetime.datetime.strptime (report['endtime'], '%Y-%m-%d %H:%M:%S')
            if report['r_starttime'] == '' or report['r_endtime'] == '':
                self.r_starttime = ''
                self.r_endtime = ''
                r_timetaken = 0
            else:
                self.r_starttime = datetime.datetime.strptime (report['r_starttime'], '%Y-%m-%d %H:%M:%S')
                self.r_endtime = datetime.datetime.strptime (report['r_endtime'], '%Y-%m-%d %H:%M:%S')
                r_timetaken = (self.r_endtime - self.r_starttime).total_seconds ()
            self.outputlocation = report['outputlocation']
            self.status = 'SUCCESS'
            print ('probe_status (complete):', self.id, self.version, self.scheduletime, self.starttime, self.endtime, self.resourceid, 'success')
            return True, self.starttime, self.endtime, 'SUCCESS', r_timetaken
        elif status == 'FAILED':
            self.status = 'FAILED'
            self.starttime = datetime.datetime.strptime (report['starttime'], '%Y-%m-%d %H:%M:%S')
            self.endtime = datetime.datetime.strptime (report['endtime'], '%Y-%m-%d %H:%M:%S')
            if report['r_starttime'] == '' or report['r_endtime'] == '':
                self.r_starttime = ''
                self.r_endtime = ''
                r_timetaken = 0
            else:
                self.r_starttime = datetime.datetime.strptime (report['r_starttime'], '%Y-%m-%d %H:%M:%S')
                self.r_endtime = datetime.datetime.strptime (report['r_endtime'], '%Y-%m-%d %H:%M:%S')
                r_timetaken = (self.r_endtime - self.r_starttime).total_seconds ()
            print ('probe status (complete):', self.id, self.version, self.scheduletime, self.starttime, self.endtime, self.resourceid, 'failed')
            return True, None, None, 'FAILED', r_timetaken
        elif status == 'CANCELLED':
            self.status = 'CANCELLED'
            self.starttime = datetime.datetime.strptime (report['starttime'], '%Y-%m-%d %H:%M:%S')
            self.endtime = datetime.datetime.strptime (report['endtime'], '%Y-%m-%d %H:%M:%S')
            if report['r_starttime'] == '' or report['r_endtime'] == '':
                self.r_starttime = ''
                self.r_endtime = ''
                r_timetaken = 0
            else:
                self.r_starttime = datetime.datetime.strptime (report['r_starttime'], '%Y-%m-%d %H:%M:%S')
                self.r_endtime = datetime.datetime.strptime (report['r_endtime'], '%Y-%m-%d %H:%M:%S')
                r_timetaken = (self.r_endtime - self.r_starttime).total_seconds ()
            print ('probe status (complete):', self.id, self.version, self.scheduletime, self.starttime, self.endtime, self.resourceid, 'cancelled')
            return True, None, None, 'CANCELLED', r_timetaken
