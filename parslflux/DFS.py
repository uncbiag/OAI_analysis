import datetime

from parslflux.resources import ResourceManager, Resource
from parslflux.input import InputManager2
from parslflux.pipeline import PipelineManager
from parslflux.workitem import WorkItem

from parslflux.scheduling_policy import Policy

class DFS (Policy):

    def remove_complete_workitem_whole (self, resource):
        #print ('remove_complete_workitem ():', resource.id)

        resource.pop_if_complete_whole ()

    def add_new_workitems (self, rmanager, imanager, pmanager, empty_resources):
        #print ('add_new_workitems (): ')

        for resource in empty_resources:
            new_workitem = self.create_workitem_full (imanager, pmanager, resource.id)
            if new_workitem != None:
                resource.add_workitem_full (new_workitem)
                new_workitem.print_data ()
