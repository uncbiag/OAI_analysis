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

        pending_workitem = scheduling_policy.get_pending_workitem_indexed_exploration (pipelinestage)

        added = False
        for exploration_resource_id in exploration_resources:
            resource = rmanager.get_resource (exploration_resource_id, True)

            new_workitem = False
            workitem = None
            if pending_workitem == None:
                parent_pipelinestages = pipelinestage.get_parents('exec')
                if len(parent_pipelinestages) <= 0:
                    workitem = scheduling_policy.get_new_workitem (pipelinestage)
                    new_workitem = True
            else:
                workitem = pending_workitem

            if workitem != None:
                if resource.get_active () == True:
                    if new_workitem == True:
                        workitem.set_resource_id(resource.id)
                        resource.add_workitem(workitem, pipelinestage.resourcetype)
                        pmanager.add_executor(workitem, resource.id, self.env.now)
                        pmanager.add_workitem_queue(workitem, self.env.now)
                    else:
                        if pipelinestage.get_exploration_workitem_added_count() <= 0:
                            workitem.set_resource_id(resource.id)
                            resource.add_workitem(workitem, pipelinestage.resourcetype)
                            pmanager.add_executor(workitem, 'temp', self.env.now)
                        else:
                            workitem_copy = workitem.get_copy()
                            workitem_copy.set_resource_id (resource.id)
                            resource.add_workitem(workitem_copy, pipelinestage.resourcetype)

                    pipelinestage.set_exploration_workitem_added (resource.id, True)
                    added = True
            if added == False:
                if new_workitem == True:
                    scheduling_policy.add_back_new_workitem (pipelinestage.index)

