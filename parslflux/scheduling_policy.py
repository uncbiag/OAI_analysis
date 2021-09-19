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

    def add_back_workitem (self, resourcetype, workitem):
        print ('add_back_workitem ():')

        print (resourcetype, workitem.get_id ())
        if resourcetype == 'CPU':
            self.gpuqueue.append (workitem)
            print (self.gpuqueue)
        else:
            self.cpuqueue.append (workitem)
            print (self.cpuqueue)

    def pop_pending_workitem (self, resourcetype):
        print ('pop_pending_workitem ():', resourcetype)
        if resourcetype == 'CPU':
            if len (self.gpuqueue) > 0:
                print (self.gpuqueue)
                item = self.gpuqueue.pop(0)
                print (self.gpuqueue)
                return item
            else:
                print ('None')
                return None
        else:
            if len (self.cpuqueue) > 0:
                print (self.cpuqueue)
                item = self.cpuqueue.pop(0)
                print (self.cpuqueue)
                return item
            else:
                print ('None')
                return None

    def get_pending_workitems (self, resourcetype):
        print ('get_pending_workitems ():', resourcetype)

        if resourcetype == 'CPU':
            return self.gpuqueue.copy ()
        else:
            return self.cpuqueue.copy ()

    def get_pending_workitems_count (self, resourcetype):
        print ('get_pending_workitems_count ():')
        if resourcetype == 'CPU':
            print (resourcetype, len (self.cpuqueue))
            return len (self.gpuqueue)

        if resourcetype == 'GPU':
            print (resourcetype, len (self.gpuqueue))
            return len (self.cpuqueue)

    def create_workitem (self, imanager, pmanager, resource_id, resourcetype):
        print ('create_workitem ():', resourcetype)

        pipelinestages = pmanager.get_pipelinestages (None, resourcetype)
        if pipelinestages == None:
            print ('None')
            return None

        if imanager.get_remaining_count () == 0:
            print ('None')
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


    def sort_complete_workitems_by_earliest_schedule_time (self, resourcetype):
        if resourcetype == 'CPU':
            self.gpuqueue = sorted (self.gpuqueue, key=lambda x:x.scheduletime)
            print (self.gpuqueue)
        else:
            self.cpuqueue = sorted (self.cpuqueue, key=lambda x:x.scheduletime)
            print (self.cpuqueue)

    def sort_complete_workitems_by_earliest_finish_time (self, resourcetype):
        if resourcetype == 'CPU':
            self.gpuqueue = sorted (self.gpuqueue, key=lambda x:x.endtime)
            print (self.gpuqueue)
        else:
            self.cpuqueue = sorted (self.cpuqueue, key=lambda x:x.endtime)
            print (self.cpuqueue)

    def sort_complete_workitems_by_latest_finish_time (self, resourcetype):
        if resourcetype == 'CPU':
            self.gpuqueue = sorted (self.gpuqueue, key=lambda x:x.endtime, reverse=True)
            print (self.gpuqueue)
        else:
            self.cpuqueue = sorted (self.cpuqueue, key=lambda x:x.endtime, reverse=True)
            print (self.cpuqueue)

class FastCompleteFirstServe2 (Policy):
    def add_new_workitems (self, rmanager, imanager, pmanager, empty_resources, resourcetype):
        print ('add_new_workitems ():')

        rankings = {}

        workitems_needed = len (empty_resources)

        workitems = []
        workitems_dict = {}
        pending_workitems_dict = {}

        print (workitems_needed)

        print (empty_resources)

        pending_workitems = self.get_pending_workitems (resourcetype)

        print (pending_workitems)


        total_done = 0

        for pending_workitem in pending_workitems:
            print (pending_workitem)
            self.pop_pending_workitem (resourcetype)
            next_workitem = pending_workitem.compose_next_workitem (pmanager, None, resourcetype)
            if next_workitem != None:
                workitems.append (next_workitem)
                workitems_dict[next_workitem.get_id ()] = next_workitem
                total_done += 1
                pending_workitems_dict[next_workitem.get_id ()] = pending_workitem

        for workitem in workitems:
            workitem_pipelinestages = workitem.get_pipelinestages ()
            encoded_workitem_pipelinestages = pmanager.encode_pipeline_stages (workitem_pipelinestages)

            exectimes = {}
            for resource in empty_resources:
                exectimes[resource.id] = resource.get_exectime (encoded_workitem_pipelinestages)

            sorted_exec_times = dict(sorted(exectimes.items(), key=lambda item: item[1]))

            rankings[workitem.get_id ()] = sorted_exec_times


        print (rankings)

        if total_done < workitems_needed:
            additional_workitems = []

            while total_done < workitems_needed:
                new_workitem = self.create_workitem (imanager, pmanager, None, resourcetype)
                if new_workitem != None:
                    additional_workitems.append (new_workitem)
                    workitems_dict[new_workitem.get_id ()] = new_workitem
                    total_done += 1
                else:
                    break

            for workitem in additional_workitems:
                workitem_pipelinestages = workitem.get_pipelinestages ()
                encoded_workitem_pipelinestages = pmanager.encode_pipeline_stages (workitem_pipelinestages)

                exectimes = {}
                for resource in empty_resources:
                    exectimes[resource.id] = resource.get_exectime (encoded_workitem_pipelinestages)

                sorted_exec_times = dict(sorted(exectimes.items(), key=lambda item: item[1]))

                rankings[workitem.get_id ()] = sorted_exec_times

        if total_done < workitems_needed:
            workitems_needed = total_done

        print (rankings)

        print (workitems_needed)

        resources_done = {}
        workitems_done = {}

        for empty_resource in empty_resources:
            resources_done[empty_resource.id] = False

        for workitem_id in workitems_dict:
            workitems_done[workitem_id] = False

        total_workitems_done = 0

        while True:
            best_rankings = {}

            for workitem_id in rankings:
                print (workitem_id)

                if workitems_done[workitem_id] == True:
                    print (workitem_id, 'done')
                    continue

                resource_rankings = rankings[workitem_id]

                print (resource_rankings)

                index = 0
                for resource_id in resource_rankings:
                    if resources_done[resource_id] == True:
                        print (resource_id, 'done')
                        continue

                    resource = rmanager.get_resource (resource_id)

                    if resource in empty_resources:
                        best_rankings[workitem_id] = [index, resource_id]
                        break
                    index += 1

            print (best_rankings)

            if len (best_rankings) == 0:
                print ('all done')
                break

            best_ranking = None
            best_ranking_workitem_id = None

            for workitem_id in best_rankings:
                ranking = best_rankings[workitem_id][0]

                if best_ranking == None:
                    best_ranking = best_rankings[workitem_id]
                    best_ranking_workitem_id = workitem_id
                elif best_ranking[0] > ranking:
                    best_ranking = best_rankings[workitem_id]
                    best_ranking_workitem_id = workitem_id

            workitem = workitems_dict[best_ranking_workitem_id]
            workitem.set_resource_id (best_ranking[1])
            resource = rmanager.get_resource (best_ranking[1])
            resource.add_workitem (workitem, resourcetype)

            workitems_done[best_ranking_workitem_id] = True
            resources_done[best_ranking[1]] = True

            total_workitems_done += 1

            if total_workitems_done == workitems_needed:
                break

        for workitem_id in rankings:
            if workitems_done[workitem_id] == True:
                continue
            workitem = workitems_dict[workitem_id]
            if workitem.get_id () in pending_workitems_dict:
                pending_workitem = pending_workitems_dict[workitem.get_id ()]
                self.add_back_workitem (resourcetype, pending_workitem)
                
class FastCompleteFirstServe (Policy):
    def add_new_workitems (self, rmanager, imanager, pmanager, empty_resources, resourcetype):
        print ('add_new_workitems ():')

        rankings = {}
        done = {}
        for resource in empty_resources:
            done[resource.id] = False
        total_done = 0

        workitems_needed = len (empty_resources)

        workitems = []
        workitems_dict = {}
        pending_workitems_dict = {}

        print (empty_resources)

        pending_workitems = self.get_pending_workitems (resourcetype)

        print (pending_workitems)

        for pending_workitem in pending_workitems:
            print (pending_workitem)
            self.pop_pending_workitem (resourcetype)
            next_workitem = pending_workitem.compose_next_workitem (pmanager, None, resourcetype)
            if next_workitem != None:
                workitems.append (next_workitem)
                workitems_dict[next_workitem.get_id ()] = next_workitem
                total_done += 1
                pending_workitems_dict[next_workitem.get_id ()] = pending_workitem

        for workitem in workitems:
            workitem_pipelinestages = workitem.get_pipelinestages ()
            encoded_workitem_pipelinestages = pmanager.encode_pipeline_stages (workitem_pipelinestages)

            exectimes = {}
            for resource in empty_resources:
                exectimes[resource.id] = resource.get_exectime (encoded_workitem_pipelinestages)

            sorted_exec_times = dict(sorted(exectimes.items(), key=lambda item: item[1]))

            rankings[workitem.get_id ()] = sorted_exec_times


        print (rankings)

        if total_done < workitems_needed:
            additional_workitems = []

            while total_done < workitems_needed:
                new_workitem = self.create_workitem (imanager, pmanager, None, resourcetype)
                if new_workitem != None:
                    additional_workitems.append (new_workitem)
                    workitems_dict[new_workitem.get_id ()] = new_workitem
                    total_done += 1
                else:
                    break

            for workitem in additional_workitems:
                workitem_pipelinestages = workitem.get_pipelinestages ()
                encoded_workitem_pipelinestages = pmanager.encode_pipeline_stages (workitem_pipelinestages)

                exectimes = {}
                for resource in empty_resources:
                    exectimes[resource.id] = resource.get_exectime (encoded_workitem_pipelinestages)

                sorted_exec_times = dict(sorted(exectimes.items(), key=lambda item: item[1]))

                rankings[workitem.get_id ()] = sorted_exec_times

        sorted_rankings = dict (sorted (rankings.items(), key = lambda item:list (item[1].values ())[0], reverse=True))

        print (sorted_rankings)

        count = 0
        for workitem_id in sorted_rankings:
            sorted_exec_times = sorted_rankings[workitem_id]

            for resource_id in sorted_exec_times:
                if done[resource_id] == True:
                    continue
                else:
                    print (workitem_id, resource_id, sorted_exec_times)
                    workitem = workitems_dict[workitem_id]
                    workitem.set_resource_id (resource_id)
                    resource = rmanager.get_resource (resource_id)
                    resource.add_workitem (workitem, resourcetype)
                    done[resource_id] = True
                    count += 1
                    break

            if count == workitems_needed:
                break

        sorted_rankings_list = list (sorted_rankings.keys ())

        for workitem_id in sorted_rankings_list [count:]:
            workitem = workitems_dict[workitem_id]
            if workitem.get_id () in pending_workitems_dict:
                pending_workitem = pending_workitems_dict[workitem.get_id ()]
                self.add_back_workitem (resourcetype, pending_workitem)
                

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

        self.sort_complete_workitems_by_earliest_schedule_time (resourcetype)
        #self.sort_complete_workitems_by_earliest_finish_time (resourcetype)
        #self.sort_complete_workitems_by_latest_finish_time (resourcetype)

        for resource_id in sorted_completion_times.keys ():
            print (resource_id, resourcetype)
            item_added = False

            pending_workitem = self.pop_pending_workitem (resourcetype)

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
