import datetime

from parslflux.resources import ResourceManager, Resource
from parslflux.input import InputManager2
from parslflux.pipeline import PipelineManager
from parslflux.workitem import WorkItem

class Policy:
    def __init__ (self, name):
        self.name = name
        self.cpuqueue = []
        self.gpuqueue = []
        self.resubmitcpuqueue = []
        self.resubmitgpuqueue = []

    def add_workitems (self, rmanager, imanager, pmanager, empty_resources, resourcetype):
        pass

    def get_pending_workitem (self, resourcetype):
        print ('get_pending_workitem ():')
        if resourcetype == 'CPU':
            if len (self.gpuqueue) > 0:
                return self.gpuqueue.pop(0)
            else:
                print ('None')
                return None
        else:
            if len (self.cpuqueue) > 0:
                return self.cpuqueue.pop(0)
            else:
                print ('None')
                return None

    def get_pending_workitems_count (self, resourcetype):
        print ('get_pending_workitems_count ():')
        if resourcetype == 'CPU':
            return len (self.cpuqueue)

        if resourcetype == 'GPU':
            return len (self.gpuqueue)

    def create_workitem (self, imanager, pmanager, resource_id, resourcetype):
        print ('create_workitem ():')

        pipelinestages = pmanager.get_pipelinestages (None, resourcetype)
        if pipelinestages == None:
            return None

        if imanager.get_remaining_count () == 0:
            return None

        images = imanager.get_images (1)

        image_key = list (images.keys())[0]

        new_workitem = WorkItem (image_key, images[image_key], None, \
                                 pipelinestages, resource_id, resourcetype, \
                                 0, '')

        return new_workitem

    def remove_complete_workitem (self, resource):
        print ('remove_complete_workitem ():', resource.id)
        cpu_workitem = resource.pop_if_complete ('CPU')

        if cpu_workitem != None:
            print (cpu_workitem.print_data ())
            if cpu_workitem.get_status () == 'SUCCESS':
                print ('adding to cpuqueue')
                self.cpuqueue.append (cpu_workitem)
            else:
                print ('adding to resubmitcpuqueue')
                self.resubmitcpuqueue.append (cpu_workitem)

        gpu_workitem = resource.pop_if_complete ('GPU')

        if gpu_workitem != None:
            print (gpu_workitem.print_data ())
            if gpu_workitem.get_status () == 'SUCCESS':
                print ('adding to gpuqueue')
                self.gpuqueue.append (gpu_workitem)
            else:
                print ('adding to resubmitgpuqueue')
                self.resubmitgpuqueue.append (gpu_workitem)

class FastCompleteFirstServe (Policy):
    def add_new_workitems (self, rmanager, imanager, pmanager, empty_resources, resourcetype):
        print ('add_new_workitems ():')

        workitems_needed = len (empty_resources)

        pending_workitems_count = self.get_pending_workitems_count (resourcetype)

        total_done = 0

        workitems = []
        workitems_dict = {}

        while total_done < workitems_needed:
            if total_done < pending_workitems_count:
                pending_workitem = self.get_pending_workitem (resourcetype)
                next_workitem = pending_workitem.compose_next_workitem (pmanager, None, resourcetype)
                if next_workitem != None:
                    workitems.append (next_workitem)
                    workitems_dict[workitem.get_id ()] = next_workitem
                    total_done += 1
                else:
                    pending_workitems_count -= 1
            else:
                new_workitem = self.create_workitem (imanager, pmanager, resource_id, resourcetype)
                if new_workitem != None:
                    workitems.append (new_workitem)
                    workitems_dict[workitem.get_id ()] = new_workitem
                    total_done += 1
                else:
                    break

        rankings = {}

        done = {}
        for resource in empty_resources:
            done[resource.id] = False

        #sort workitems by 

        for workitem in workitems:
            workitem_pipelinestages = workitem.get_pipelinestages ()
            encoded_workitem_pipelinestages = pmanager.encode_pipeline_stages (self.pipelinestages)

            exectimes = {}
            for resource in empty_resources:
                exectimes[resource.id] = resource.get_exec_time (encoded_workitem_pipelinestages)

            sorted_exec_times = dict(sorted(exectimes.items(), key=lambda item: item[1]))

            rankings[workitem.get_id ()] = sorted_exec_times

        sorted_rankings = dict (sorted (rankings.items(), key = lambda item:list (item[1].values ())[0]))


        for ranking in sorted_rankings:
            

        for resource in empty_resources:
            completion_time = resource.get_

class FirstCompleteFirstServe (Policy):

    def add_new_workitems (self, rmanager, imanager, pmanager, empty_resources, resourcetype):
        print ('add_new_workitems ():')
        completion_times = {}

        for resource in empty_resources:
            completion_time = resource.get_last_completion_time (resourcetype)

            if completion_time == None:
                completion_times[resource.id] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            else:
                completion_times[resource.id] = completion_time

        sorted_completion_times = dict(sorted(completion_times.items(), key=lambda item: item[1]))

        print (sorted_completion_times)

        for resource_id in sorted_completion_times.keys ():
            print (resource_id, resourcetype)
            item_added = False

            pending_workitem = self.get_pending_workitem (resourcetype)

            resource = rmanager.get_resource (resource_id)

            if pending_workitem != None:
                pending_workitem.print_data ()
                next_workitem = pending_workitem.compose_next_workitem (pmanager, resource_id, resourcetype)
                if next_workitem != None:
                    resource.add_workitem (next_workitem, resourcetype)
                    next_workitem.print_data()
                    item_added = True

            if item_added == False:
                new_workitem = self.create_workitem (imanager, pmanager, resource_id, resourcetype)

                if new_workitem != None:
                    resource.add_workitem (new_workitem, resourcetype)
                    new_workitem.print_data ()
                    item_added = True

            if item_added == False:
                print ('add_workitems ()', resource_id, 'workitems not available')
                break
