from parslflux.pipeline import PipelineManager
import datetime
import flux

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
        print ('print_data ()', self.id, self.version, self.resourceid)

    def submit (self, pmanager):
        workitem = {}
        workitem['pipelinestages'] = pmanager.encode_pipeline_stages(self.pipelinestages)
        workitem['id'] = self.id
        workitem['version'] = str(self.version)
        workitem['data'] = self.data
        workitem['resourcetype'] = self.resourcetype
        workitem['inputlocation'] = self.inputlocation
        workitem['collectfrom'] = self.collectfrom
        workitem['workerid'] = self.resourceid
        self.scheduletime = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        self.scheduletime = datetime.datetime.strptime (self.scheduletime, '%Y-%m-%d %H:%M:%S')

        print ('submit (): ', workitem)

        f = flux.Flux ()
        f.rpc (b"parslmanager.workitem.submit", workitem)

        self.status = 'SCHEDULED'

    def probe_status (self):
        print ('probe_status ():', self.id, self.version)
        f = flux.Flux ()
        r = f.rpc ("parslmanager.workitem.status", {"workerid":self.resourceid, \
                   "id":self.id, "version":str(self.version)}).get ()

        print ('report', r)

        report = r['report']

        if type (report) is not dict and report == 'empty':
            print ('empty report')
            return False, None, None

        status = report['status']

        self.iscomplete = True

        if status == 'SUCCESS':
            self.starttime = datetime.datetime.strptime (report['starttime'], '%Y-%m-%d %H:%M:%S')
            self.endtime = datetime.datetime.strptime (report['endtime'], '%Y-%m-%d %H:%M:%S')
            self.outputlocation = report['outputlocation']
            self.status = 'SUCCESS'
            print ('probe_status (complete):', self.id, self.version, self.scheduletime, self.starttime, self.endtime, self.resourceid)
            return True, self.starttime, self.endtime
        else:
            self.status = 'FAILURE'
            return True, None, None
