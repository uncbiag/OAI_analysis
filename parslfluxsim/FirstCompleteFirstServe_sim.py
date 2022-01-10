import datetime

from parslfluxsim.resources_sim import ResourceManager, Resource
from parslflux.input import InputManager2
from parslflux.pipeline import PipelineManager
from parslfluxsim.workitem_sim import WorkItem

from parslfluxsim.scheduling_policy_sim import Policy

class FirstCompleteFirstServe (Policy):

    def __init__(self, name, env):
        Policy.__init__(self, name)
        self.env = env

    def add_new_workitems (self, rmanager, imanager, pmanager, empty_resources, resourcetype):
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

        self.sort_complete_workitems_by_stage_id (resourcetype)
        #self.sort_complete_workitems_by_earliest_schedule_time (resourcetype)
        #self.sort_complete_workitems_by_earliest_finish_time (resourcetype)
        #self.sort_complete_workitems_by_latest_finish_time (resourcetype)

        for resource_id in sorted_completion_times.keys ():
            #print (resource_id, resourcetype)
            item_added = False
            resource = rmanager.get_resource (resource_id)

            resubmit_workitem = self.pop_resubmit_workitem (resourcetype)

            if resubmit_workitem != None:
                resubmit_workitem.print_data ()
                resubmit_workitem.set_resource_id (resource_id)
                resource.add_workitem (resubmit_workitem, resourcetype)
                item_added = True

            if item_added == False:
                pending_workitem = self.pop_pending_workitem (resourcetype)

                if pending_workitem != None:
                    pending_workitem.print_data ()
                    next_workitem = pending_workitem.compose_next_workitem (pmanager, resource_id, resourcetype)
                    if next_workitem != None:
                        resource.add_workitem (next_workitem, resourcetype)
                        pmanager.add_executor (next_workitem, resource, self.env.now)
                        next_workitem.print_data()
                        item_added = True

            if item_added == False:
                #new_workitem = self.create_workitem (imanager, pmanager, resource_id, resourcetype)
                new_workitem = self.get_new_workitem(resourcetype)

                if new_workitem != None:
                    new_workitem.set_resource_id (resource_id)
                    resource.add_workitem (new_workitem, resourcetype)
                    pmanager.add_workitem_queue(new_workitem, self.env.now)
                    pmanager.add_executor (new_workitem, resource, self.env.now)
                    new_workitem.print_data ()
                    item_added = True

            if item_added == False:
                #print ('add_workitems ()', resource_id, 'workitems not available')
                break
