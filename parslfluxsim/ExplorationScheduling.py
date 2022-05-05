import datetime

from parslfluxsim.resources_sim import ResourceManager, Resource
from parslflux.input import InputManager2
from parslflux.pipeline import PipelineManager
from parslfluxsim.workitem_sim import WorkItem
import copy

from parslfluxsim.scheduling_policy_sim import Policy

class ExplorationScheduling (Policy):

    def __init__(self, name, env, pmanager):
        Policy.__init__(self, name, pmanager)
        self.env = env

    def add_workitem_exploration (self, rmanager, pmanager, pipelinestage, scheduling_policy):
        exploration_resources = pipelinestage.get_explorers ()

        new_workitem = False
        scheduling_policy.sort_complete_workitems_by_earliest_finish_time (pipelinestage.resourcetype)
        workitem = scheduling_policy.pop_pending_workitem_indexed (pipelinestage.resourcetype, pipelinestage.index)

        if workitem == None:
            workitem = scheduling_policy.get_new_workitem (pipelinestage.resourcetype)
            new_workitem = True
        else:
            print ('add_workitem_exploration ()', 'pending')
            workitem.print_data ()

        print ('add_workitem_exploration', pipelinestage.name, workitem.id)

        if workitem != None:
            added = False
            for exploration_resource_id in exploration_resources:
                resource = rmanager.get_resource (exploration_resource_id, True)

                if resource.get_active () == True:
                    if pipelinestage.get_exploration_workitem_added_count() <= 0:
                        workitem.set_resource_id(resource.id)
                        resource.add_workitem(workitem, pipelinestage.resourcetype)

                        pmanager.add_executor(workitem, 'temp', self.env.now)
                        if new_workitem == True:
                            pmanager.add_workitem_queue(workitem, self.env.now)
                            workitem.print_data()
                    else:
                        workitem_copy = workitem.get_copy()
                        workitem_copy.set_resource_id (resource.id)
                        resource.add_workitem(workitem_copy, pipelinestage.resourcetype)

                    pipelinestage.set_exploration_workitem_added (resource.id, True)
                    added = True
            if added == False:
                if new_workitem == False:
                    scheduling_policy.add_back_workitem (pipelinestage.resourcetype, workitem)
                else:
                    scheduling_policy.add_back_new_workitem (workitem)
                #else:
                #    print ('add_workitem_exploration ()', exploration_resource_id, 'not active yet')

