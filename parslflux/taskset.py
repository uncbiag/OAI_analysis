import copy
import json
import flux, time
from flux.job import JobspecV1
from flux.core.inner import ffi, raw

class Taskset:
    def __init__ (self, tasksetid, resourcetype, iteration):
        self.resourcetype = resourcetype
        self.tasksetid = str(tasksetid)
        self.input = {}
        self.pipelinestages = []
        self.inputsize = 0
        self.endofpipeline = False
        self.complete = False
        self.iteration = str (iteration)

    def print_data(self):
        print ('tasksetid: ', self.tasksetid)
        pipelinestages = []
        for pipelinestage in self.pipelinestages:
            pipelinestages.append(pipelinestage.name)
        print('pipline stages: ', pipelinestages)
        print ('input size: ', self.inputsize)
        print ('input-resource map: ')
        print ('#################')
        for key in self.input.keys():
            mapping = self.input[key]
            print ('resource id:', key)
            print ('count: ', mapping['count'])
            print ('images:')
            images = mapping['images']
            for key1 in images.keys():
                image = images[key1]
                print ('id: ', key1, 'data', image)
                print('-------------------')

    def get_resource_type (self):
        return self.resourcetype

    def set_endofpipeline (self, endofpipeline):
        self.endofpipeline = endofpipeline

    def get_endofpipeline (self):
        return self.endofpipeline

    def get_taskset_len (self):
        return len(self.pipelinestages)

    def add_pipelinestage (self, pipelinestage):
        self.pipelinestages.append (pipelinestage)

    def add_input (self, rmanager, imanager, pmanager, resource):

        print ('add_input ():', resource.id)

        if self.resourcetype == 'CPU' and resource.cputype == False:
            return
        if self.resourcetype == 'GPU' and resource.gputype == False:
            return

        images = imanager.get_images (resource.get_chunksize (self.resourcetype,
                                      pmanager.encode_pipeline_stages(self.pipelinestages)))

        if len(images) > 0:
            self.input[resource.id] = {}
            self.input[resource.id]['images'] = {}
            self.input[resource.id]['count'] = len(images)
            self.input[resource.id]['complete'] = 0
            self.input[resource.id]['scheduled'] = 0
            self.input[resource.id]['status'] = 'QUEUED'
            for image_id in images.keys():
                self.input[resource.id]['images'][image_id] = {}
                self.input[resource.id]['images'][image_id]['data'] = images[image_id]
                self.input[resource.id]['images'][image_id]['collectfrom'] = resource.id
                self.input[resource.id]['images'][image_id]['status'] = 'QUEUED'
                self.input[resource.id]['images'][image_id]['reallocated'] = False
            self.inputsize += self.input[resource.id]['count']
            return 0
        else:
            return -1

    def add_input_taskset (self, input_taskset, rmanager, pmanager, resource):
        print ('add_input_taskset ():', resource.id)

        if self.resourcetype == 'CPU' and resource.cputype == False:
            return
        if self.resourcetype == 'GPU' and resource.gputype == False:
            return

        chunksize = resource.get_chunksize(self.resourcetype,
                                           pmanager.encode_pipeline_stages(self.pipelinestages))

        #first collect it from its own resource.id
        resource_input_images = input_taskset.input[resource.id]

        self.input[resource.id] = {}
        self.input[resource.id]['count'] = 0
        self.input[resource.id]['images'] = {}
        self.input[resource.id]['status'] = 'SCHEDULED'
        self.input[resource.id]['complete'] = 0
        self.input[resource.id]['scheduled'] = 0

        if resource_input_images['count'] > 0:
            index = 0
            image_keys = list (resource_input_images['images'].keys())
            while index < resource_input_images['count'] and self.input[resource.id]['count'] < chunksize:
                print (resource_input_images)
                print (image_keys)
                if resource_input_images['images'][image_keys[index]]['reallocated'] == False:
                    self.input[resource.id]['images'][image_keys[index]] = {}
                    self.input[resource.id]['images'][image_keys[index]]['data'] = resource_input_images['images'][image_keys[index]]['data']
                    self.input[resource.id]['images'][image_keys[index]]['collectfrom'] = resource.id
                    self.input[resource.id]['images'][image_keys[index]]['status'] = 'QUEUED'
                    self.input[resource.id]['images'][image_keys[index]]['reallocated'] = False
                    resource_input_images['images'][image_keys[index]]['reallocated'] = True
                    self.input[resource.id]['count'] += 1
                index += 1

        if self.input[resource.id]['count'] < chunksize:
            #search for the rest of the images
            resource_keys = list (input_taskset.input.keys ())

            for resource_key in resource_keys:
                if resource_key == resource.id:
                    continue

                resource_input_images = input_taskset.input[resource_key]['images']
                if resource_input_images['count'] > 0:
                    index = 0
                    image_keys = list (resource_input_images['images'].keys())
                    while index < resource_input_images['count'] and self.input[resource.id]['count'] < chunksize:
                        if resource_input_images['images'][image_keys[index]]['reallocated'] == False:
                            self.input[resource.id]['images'][image_keys[index]] = {}
                            self.input[resource.id]['images'][image_keys[index]]['data'] = resource_input_images['images'][image_keys[index]]['data'] 
                            self.input[resource.id]['images'][image_keys[index]]['collectfrom'] = resource_key
                            self.input[resource.id]['images'][image_keys[index]]['status'] = 'QUEUED'
                            self.input[resource.id]['images'][image_keys[index]]['reallocated'] = False
                            resource_input_images['images'][image_keys[index]]['reallocated'] = True
                            self.input[resource.id]['count'] += 1
                        index += 1

    def submit_support (self, rmanager, pmanager, resource_ids):
        print ('submitting support taskset', self.tasksetid)
        taskset = {}
        taskset['pipelinestages'] = pmanager.encode_pipeline_stages(self.pipelinestages)
        taskset['id'] = self.tasksetid
        taskset['resourcetype'] = self.resourcetype
        taskset['input'] = {}
        taskset['iteration'] = self.iteration
        if len (resource_ids) == 0:
            resource_ids = self.input.keys ()

        for resource_id in resource_ids:
            taskset['input'][str(resource_id)] = {}
            taskset['input'][str(resource_id)]['count'] = self.input[resource_id]['count']
            taskset['input'][str(resource_id)]['images'] = self.input[resource_id]['images']

            self.input[resource_id]['status'] = 'SCHEDULED'
            self.input[resource_id]['scheduled'] = self.input[resource_id]['count']

            images = self.input[resource_id]['images']

            for image_id in images.keys ():
                self.input[resource_id]['images'][image_id]['status'] = 'SCHEDULED'

        print (taskset)

        f = flux.Flux ()
        r = f.rpc(b"parslmanager.taskset.submit", taskset)

        for resource_id in resource_ids:
            resource = rmanager.get_resource(resource_id)
            resource.set_support_taskset (self.iteration, self.tasksetid, self.resourcetype)
            resource.set_current_taskset (self.resourcetype, 'SUPPORT')
        return ""

    def submit_main (self, rmanager, pmanager, resource_ids):
        print ('submitting main taskset', self.tasksetid)
        taskset = {}
        taskset['pipelinestages'] = pmanager.encode_pipeline_stages(self.pipelinestages)
        taskset['id'] = self.tasksetid
        taskset['resourcetype'] = self.resourcetype
        taskset['input'] = {}
        taskset['iteration'] = self.iteration
        if len (resource_ids) == 0:
            resource_ids = self.input.keys ()

        for resource_id in resource_ids:
            taskset['input'][str(resource_id)] = {}
            taskset['input'][str(resource_id)]['count'] = self.input[resource_id]['count']
            taskset['input'][str(resource_id)]['images'] = self.input[resource_id]['images']
            self.input[resource_id]['status'] = 'SCHEDULED'
            self.input[resource_id]['scheduled'] = self.input[resource_id]['count']

            images = self.input[resource_id]['images']
            for image_id in images.keys ():
                self.input[resource_id]['images'][image_id]['status'] = 'SCHEDULED'

        print (taskset)

        f = flux.Flux ()
        f.rpc(b"parslmanager.taskset.submit", taskset)

        for resource_id in resource_ids:
            resource = rmanager.get_resource(resource_id)
            resource.set_main_taskset (self.iteration, self.tasksetid, self.resourcetype)
            resource.set_current_taskset (self.resourcetype, 'MAIN')

        return ""

    def get_status (self):
        f = flux.Flux ()

        print ('getting status')

        r = f.rpc("parslmanager.taskset.status", {"tasksetid": self.tasksetid}).get()

        print (r)

        report = r['report']

        if type (report) is not dict and report == 'empty':
            print ('empty report')
            return

        for workerid in report.keys ():
            if type (report[workerid]) is not list:
                if report[workerid] == 'INCOMPLETE':
                    print (workerid, 'incomplete')
                    continue
                elif report[workerid] == 'FAILED':
                    print (workerid, 'failed')
                    self.input[workerid]['status'] = 'FAILED'
            else:
                print (workerid, 'success')
                self.input[workerid]['times'] = report[workerid]
                self.input[workerid]['complete'] = self.input[workerid]['count']
                self.input[workerid]['scheduled'] = 0
