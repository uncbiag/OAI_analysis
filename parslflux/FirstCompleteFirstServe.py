import datetime

from parslflux.resources import ResourceManager, Resource
from parslflux.input import InputManager2
from parslflux.pipeline import PipelineManager
from parslflux.workitem import WorkItem

from parslflux.scheduling_policy import Policy

class FirstCompleteFirstServe (Policy):

    def add_new_workitems (self, rmanager, imanager, pmanager, empty_resources, resourcetype):
        print ('add_new_workitems ():')
        completion_times = {}

        for resource in empty_resources:
            completion_time = resource.get_last_completion_time (resourcetype)

            if completion_time == None:
                completion_times[resource.id] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                completion_times[resource.id] = datetime.datetime.strptime (completion_times[resource.id], '%Y-%m-%d %H:%M:%S')
            else:
                completion_times[resource.id] = completion_time

        print (completion_times)

        sorted_completion_times = dict(sorted(completion_times.items(), key=lambda item: item[1]))

        print (sorted_completion_times)

        self.sort_complete_workitems_by_earliest_schedule_time (resourcetype)
        #self.sort_complete_workitems_by_earliest_finish_time (resourcetype)
        #self.sort_complete_workitems_by_latest_finish_time (resourcetype)

        for resource_id in sorted_completion_times.keys ():
            print (resource_id, resourcetype)
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
