import datetime

from parslfluxsim.resources_sim import ResourceManager, Resource
from parslflux.input import InputManager2
from parslflux.pipeline import PipelineManager, PipelineStage
from parslfluxsim.workitem_sim import WorkItem
from parslfluxsim.bagofworkitems_sim import BagOfWorkItems

class FirstCompleteFirstServe:

    def __init__(self, env):
        self.env = env

    def remove_complete_workitem (self, resource, executed_pipelinestage, imanager):
        #print ('remove_complete_workitem ():', resource.id, len (self.cpuqueue), len (self.gpuqueue))
        workitem = resource.pop_if_complete (executed_pipelinestage.resourcetype)

        if workitem != None:
            if workitem.get_status () == 'SUCCESS':
                #print ('adding to cpuqueue')
                imanager.set_complete(workitem, True)

                data_children_pipelinestages = executed_pipelinestage.get_children('data')
                if len (data_children_pipelinestages) > 0:
                    resource.pfs.store_file(workitem, data_children_pipelinestages)

                children_pipelinestages = executed_pipelinestage.get_children ('exec')

                for child_pipelinestage in children_pipelinestages:
                    parents_of_child = child_pipelinestage.get_parents ('exec')

                    all_parents_complete = True
                    for parent_of_child in parents_of_child:
                        if parent_of_child.index == executed_pipelinestage.index:
                            continue
                        #print (executed_pipelinestage.name, child_pipelinestage.name, parent_of_child.name, imanager.is_complete (cpu_workitem.id, parent_of_child.index))
                        if imanager.is_complete (workitem.id, parent_of_child.index) == True:
                            continue
                        else:
                            all_parents_complete = False
                            break
                    if all_parents_complete == True:
                        next_workitem = workitem.compose_next_workitem (child_pipelinestage)
                        child_pipelinestage.bagofworkitems.add_workitem(next_workitem)
                        imanager.set_complete (next_workitem, False)

                return True

        return False

    def add_new_workitems_DFS_pipelinestage (self, rmanager, imanager, dmanager, empty_resources, pipelinestage):
        #print('add_new_workitems_DFS_pipelinestage', pipelinestageindex, len(self.cpuqueue), len(self.gpuqueue))
        completion_times = {}

        for resource in empty_resources:
            completion_time = resource.get_last_completion_time(pipelinestage.resourcetype)

            if completion_time == None:
                completion_times[resource.id] = self.env.now
            else:
                completion_times[resource.id] = completion_time

        sorted_completion_times = dict(sorted(completion_times.items(), key=lambda item: item[1]))

        #self.sort_complete_workitems_by_priority(resourcetype)
        # self.sort_complete_workitems_by_stage (resourcetype)
        # self.sort_complete_workitems_by_earliest_schedule_time (resourcetype)
        pipelinestage.bagofworkitems.sort_complete_workitems_by_earliest_finish_time ()
        # self.sort_complete_workitems_by_latest_finish_time (resourcetype)

        for resource_id in sorted_completion_times.keys():
            # print (resource_id, resourcetype)
            item_added = False
            resource = rmanager.get_resource(resource_id, active=True)

            workitem = pipelinestage.bagofworkitems.pop_workitem ()

            if workitem != None:
                #print('pending_workitem ()', pending_workitem.id, pending_workitem.version)
                self.set_io_transfer_stats (rmanager, imanager, dmanager, workitem, pipelinestage, resource_id)
                workitem.set_resource_id(resource_id)
                resource.add_workitem(workitem, pipelinestage.resourcetype)
                item_added = True

            if item_added == False:
                # print ('add_workitems ()', resource_id, 'workitems not available')
                break

            print ('add_new_workitems_DFS_pipelinestage', workitem.id, pipelinestage.name, resource_id)

    def set_io_transfer_stats (self, rmanager, imanager, dmanager, workitem, pipelinestage, dest_resource_id):
        max_transfer_latency = 0

        parent_pipelinestages = pipelinestage.get_parents ('data')
        dest_resource = rmanager.get_resource(dest_resource_id, active=True)

        for parent_pipelinestage in parent_pipelinestages:
            src_resource_id = imanager.get_input_owner (workitem.id, parent_pipelinestage.index)

            print('set_id_transfer_stats ()', workitem.id, parent_pipelinestage.index, src_resource_id, dest_resource_id)

            src_resource = rmanager.get_resource (src_resource_id, active=True)

            read_latency = src_resource.pfs.read_file (workitem.id, pipelinestage.name, parent_pipelinestage.name)
            transfer_latency = dmanager.get_interdomain_transfer_latency (src_resource.domain_id,\
                                                                          dest_resource.domain_id,\
                                                                          parent_pipelinestage.output_size)

            total_latency = read_latency + transfer_latency

            if max_transfer_latency < total_latency:
                max_transfer_latency = total_latency

        workitem.input_read_time = max_transfer_latency
        workitem.output_write_time = dest_resource.pfs.get_write_latency(pipelinestage.output_size)

    def add_new_workitems_DFS (self, rmanager, imanager, pmanager, empty_resources, resourcetype):
        #print ('add_new_workitems ():')
        completion_times = {}

        for resource in empty_resources:
            completion_time = resource.get_last_completion_time (resourcetype)

            if completion_time == None:
                completion_times[resource.id] = self.env.now
            else:
                completion_times[resource.id] = completion_time

        #print (completion_times)

        sorted_completion_times = dict(sorted(completion_times.items(), key=lambda item: item[1]))

        #print (sorted_completion_times)

        self.sort_complete_workitems_by_priority(resourcetype)
        #self.sort_complete_workitems_by_stage (resourcetype)
        #self.sort_complete_workitems_by_earliest_schedule_time (resourcetype)
        #self.sort_complete_workitems_by_earliest_finish_time (resourcetype)
        #self.sort_complete_workitems_by_latest_finish_time (resourcetype)

        for resource_id in sorted_completion_times.keys ():
            #print (resource_id, resourcetype)
            item_added = False
            resource = rmanager.get_resource (resource_id, active=True)

            resubmit_workitem = self.pop_resubmit_workitem (resourcetype)

            if resubmit_workitem != None:
                resubmit_workitem.print_data ()
                resubmit_workitem.set_resource_id (resource_id)
                resource.add_workitem (resubmit_workitem, resourcetype)
                pmanager.add_executor(resubmit_workitem, resource.id, self.env.now)
                item_added = True

            if item_added == False:
                pending_workitem = self.pop_pending_workitem (resourcetype)

                if pending_workitem != None:
                    #print('pending_workitem ()', pending_workitem.id, pending_workitem.version)
                    pending_workitem.set_resource_id (resource_id)
                    resource.add_workitem (pending_workitem, resourcetype)
                    pmanager.add_executor (pending_workitem, resource.id, self.env.now)
                    item_added = True

            if item_added == False:
                #new_workitem = self.create_workitem (imanager, pmanager, resource_id, resourcetype)
                new_workitem = self.get_new_workitem(resourcetype)

                if new_workitem != None:
                    new_workitem.set_resource_id (resource_id)
                    resource.add_workitem (new_workitem, resourcetype)
                    pmanager.add_workitem_queue(new_workitem, self.env.now)
                    pmanager.add_executor (new_workitem, resource.id, self.env.now)
                    #new_workitem.print_data ()
                    item_added = True

            if item_added == False:
                #print ('add_workitems ()', resource_id, 'workitems not available')
                break
