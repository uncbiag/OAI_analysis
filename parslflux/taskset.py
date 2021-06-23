import copy
import json
import flux
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
                print ('id: ', key1, 'location: ', image['location'], 'collectfrom: ', image['collectfrom'])
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

    def add_input (self, rmanager, imanager, pmanager):
        resources = rmanager.get_resources()
        for resource in resources:
            if self.resourcetype == 'CPU' and resource.cputype == False:
                continue

            if self.resourcetype == 'GPU' and resource.gputype == False:
                continue
            self.input[resource.id] = {}
            images = imanager.get_input(resource.get_chunksize(self.resourcetype,
                                                               pmanager.encode_pipeline_stages(self.pipelinestages)))
            self.input[resource.id]['images'] = {}
            self.input[resource.id]['count'] = len(images)
            self.input[resource.id]['complete'] = 0
            self.input[resource.id]['scheduled'] = 0
            self.input[resource.id]['status'] = 'QUEUED'
            for image in images:
                print (image.get_id ())
                self.input[resource.id]['images'][image.get_id ()] = {}
                self.input[resource.id]['images'][image.get_id()]['location'] = image.get_location ()
                self.input[resource.id]['images'][image.get_id()]['collectfrom'] = resource.id
                self.input[resource.id]['images'][image.get_id()]['status'] = 'QUEUED'
            self.inputsize += self.input[resource.id]['count']

    def add_input_taskset (self, input_taskset, rmanager, pmanager):
        resources = rmanager.get_resources()

        self.inputsize = input_taskset.inputsize

        total_chunksize = 0
        for resource in resources:
            total_chunksize += resource.get_chunksize(self.resourcetype,
                                                      pmanager.encode_pipeline_stages(self.pipelinestages))
        unassigned_items = {}
        for resource in resources:
            if self.resourcetype == 'CPU' and resource.cputype == False:
                continue

            if self.resourcetype == 'GPU' and resource.gputype == False:
                continue
            self.input[resource.id] = {}
            self.input[resource.id]['images'] = {}
            self.input[resource.id]['count'] = int (resource.get_chunksize(self.resourcetype,
                                               pmanager.encode_pipeline_stages(self.pipelinestages)) / total_chunksize * self.inputsize)
            self.input[resource.id]['complete'] = 0
            self.input[resource.id]['scheduled'] = 0
            self.input[resource.id]['status'] = 'QUEUED'
            if resource.id in input_taskset.input.keys():
                images = input_taskset.input[resource.id]['images']
                index = 0
                for id in images.keys():
                    if index >= self.input[resource.id]['count']:
                        break
                    self.input[resource.id]['images'][id] = images[id]
                    self.input[resource.id]['images'][id]['location'] = 'self'
                    self.input[resource.id]['images'][id]['collectfrom'] = resource.id
                    self.input[resource.id]['images'][id]['status'] = 'QUEUED'
                    index += 1

                if self.input[resource.id]['count'] < input_taskset.input[resource.id]['count']:
                    local_unassigned_items = input_taskset.input[resource.id]['images'].values()
                    local_unassigned_keys = input_taskset.input[resource.id]['images'].keys()
                    for i in range (index, input_taskset.input[resource.id]['count']):
                        unassigned_items[local_unassigned_keys[i]] = local_unassigned_items[i]

        for resource in resources:
            index = 0
            if self.input[resource.id]['count'] > len (self.input[resource.id]['images'].keys()):
                for i in range (0, self.input[resource.id]['count'] - len (self.input[resource.resource.id]['images'].keys())):
                    self.input[resource.id]['images'][unassigned_items.keys()[index]] = unassigned_items.values()[index]
                    self.input[resource.id]['images'][unassigned_items.keys()[index]]['location'] = 'sync:' + str(unassigned_items.values()[index]['collectfrom'])
                    self.input[resource.id]['images'][unassigned_items.keys()[index]]['status'] = 'QUEUED'
                    index += 1

    def submit_support (self, rmanager, pmanager, resource_ids, count):
        taskset = {}
        taskset['pipelinestages'] = pmanager.encode_pipeline_stages(self.pipelinestages)
        taskset['id'] = self.tasksetid
        taskset['resourcetype'] = self.resourcetype
        taskset['input'] = []
        for resource_id in resource_ids:
            input = {}
            input['count'] = self.input[resource_id]['count']
            input['images'] = self.input[resource_id]['images']
            workerinput = {}
            workerinput[str(resource_id)] = input
            taskset['input'].append(workerinput)
        taskset['iteration'] = self.iteration

        f = flux.Flux()
        r = f.rpc(b"parslmanager.taskset.submit", taskset).get()

        print(r)

        for resource_id in resource_ids:
            resource = rmanager.get_resource(resource_id)
            resource.set_support_taskset (self, self.iteration, self.tasksetid, self.resourcetype)
        return ""

    def submit_main (self, rmanager, pmanager, resource_ids):
        taskset = {}
        taskset['pipelinestages'] = pmanager.encode_pipeline_stages(self.pipelinestages)
        taskset['id'] = self.tasksetid
        taskset['resourcetype'] = self.resourcetype
        taskset['input'] = {}
        taskset['iteration'] = self.iteration
        if len (resource_ids) == 0:
            for resource_id in self.input.keys():
                taskset['input'][str(resource_id)] = {}
                taskset['input'][str(resource_id)]['count'] = self.input[resource_id]['count']
                taskset['input'][str(resource_id)]['images'] = self.input[resource_id]['images']
        else:
            for resource_id in resource_ids:
                taskset['input'][str(resource_id)] = {}
                taskset['input'][str(resource_id)]['count'] = self.input[resource_id]['count']
                taskset['input'][str(resource_id)]['images'] = self.input[resource_id]['images']

        print (taskset)

        f = flux.Flux ()
        r = f.rpc(b"parslmanager.taskset.submit", taskset).get()

        print (r)

        for resource_id in resource_ids:
            resource = rmanager.get_resource(resource_id)
            resource.set_main_taskset (self.iteration, self.tasksetid, self.resourcetype)

        return ""

    def get_status (self):
        return ""
