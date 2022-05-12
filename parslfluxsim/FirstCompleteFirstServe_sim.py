import datetime

from parslfluxsim.resources_sim import ResourceManager, Resource
from parslflux.input import InputManager2
from parslflux.pipeline import PipelineManager
from parslfluxsim.workitem_sim import WorkItem

from parslfluxsim.scheduling_policy_sim import Policy

class FirstCompleteFirstServe (Policy):

    def __init__(self, name, env, pmanager):
        Policy.__init__(self, name, pmanager)
        self.env = env

    def add_new_workitems (self, rmanager, pmanager, empty_resources, resourcetype, last_phase_closed_index, perform_idle_check):
        # print ('add_new_workitems ():')
        completion_times = {}

        for resource in empty_resources:
            completion_time = resource.get_last_completion_time(resourcetype)

            if completion_time == None:
                completion_times[resource.id] = self.env.now
            else:
                completion_times[resource.id] = completion_time

        # print (completion_times)

        sorted_completion_times = dict(sorted(completion_times.items(), key=lambda item: item[1]))

        for resource_id in sorted_completion_times.keys():
            self.sort_complete_workitems_by_phase_and_stage(resourcetype)
            item_added = False
            resource = rmanager.get_resource(resource_id)

            new_workitem = self.get_new_workitem(resourcetype)

            if new_workitem != None:#make sure we process the stage 0's items first since they are not in the pending queue
                index = pmanager.check_new_workitem_index ()

                pending_workitems = self.get_pending_workitems (resourcetype)

                if len (pending_workitems) > 0 and index <= pending_workitems[0].phase_index:
                    new_workitem.set_resource_id(resource_id)
                    resource.add_workitem(new_workitem, resourcetype)
                    pmanager.add_workitem_queue(new_workitem, self.env.now)
                    pmanager.add_executor(new_workitem, resource, self.env.now)
                    # new_workitem.print_data ()
                    item_added = True
                elif len (pending_workitems) <= 0:
                    '''
                    if perform_idle_check == True:
                        ret = pmanager.check_workitem_waiting_idleness (rmanager, resource_id, resourcetype, last_phase_closed_index, new_workitem.pipelinestage, self.env.now)
                        if ret == False:
                            self.add_back_new_workitem(new_workitem)
                            return
                    '''
                    new_workitem.set_resource_id(resource_id)
                    resource.add_workitem(new_workitem, resourcetype)
                    pmanager.add_workitem_queue(new_workitem, self.env.now, index)
                    pmanager.add_executor(new_workitem, resource, self.env.now)
                    item_added = True
                else:
                    self.add_back_new_workitem (new_workitem)

            if item_added == False:
                pending_workitem = None
                if perform_idle_check == True:
                    pending_workitem_list = []
                    idle_period_end = -1
                    #print ('pending_workitems () 1', self.get_pending_workitems(resourcetype), pending_workitem)
                    while True:
                        pending_workitem = self.pop_pending_workitem(resourcetype)

                        if pending_workitem != None and pending_workitem.phase_index == last_phase_closed_index + 1:
                            ret, idle_period_ret = pmanager.check_throttle_idleness(last_phase_closed_index,
                                                                                    self.env.now, pending_workitem)

                            #print (pending_workitem.pipelinestage.name, pending_workitem.phase_index, ret )

                            if ret == False:
                                idle_period_end = idle_period_ret
                                pending_workitem_list.append(pending_workitem)
                            else:
                                idle_period_end = -1
                                self.add_back_workitem(resourcetype, pending_workitem)
                                break
                        else:
                            if pending_workitem != None:
                                self.add_back_workitem(resourcetype, pending_workitem)
                            break
                        #print ('pending_workitme_list ()', len (pending_workitem_list))

                    pending_workitem = None

                    time_left = idle_period_end - self.env.now

                    new_workitem = self.get_new_workitem(resourcetype)
                    if new_workitem != None:
                        new_workitem_exectime = resource.get_exectime(new_workitem.pipelinestage.name, new_workitem.pipelinestage.resourcetype)
                        if new_workitem_exectime > time_left and idle_period_end != -1:
                            self.add_back_new_workitem(new_workitem)
                            new_workitem = None

                    candidates = self.get_pending_workitems(resourcetype)
                    sorted_candidates = sorted (candidates, key=lambda x:(x.phase_index, x.version))

                    candidate_index = 0
                    for candidate in sorted_candidates:
                        next_pipelinestage = candidate.get_next_pipelinestage(pmanager, resourcetype)
                        exectime = resource.get_exectime(next_pipelinestage.name, new_workitem.pipelinestage.resourcetype)

                        if exectime < time_left or idle_period_end == -1:
                            break
                        candidate_index += 1

                    pending_workitem_candidate = None
                    if candidate_index < len (sorted_candidates):
                        pending_workitem_candidate = sorted_candidates[candidate_index]

                    #if pending_workitem_candidate != None:
                    #    print (pending_workitem_candidate.pipelinestage.name, pending_workitem_candidate.id, pending_workitem_candidate.phase_index, pending_workitem)

                    #if new_workitem != None:
                    #    print (new_workitem.pipelinestage.name, new_workitem.id, pmanager.check_new_workitem_index())

                    if new_workitem != None:
                        if pending_workitem_candidate != None:
                            new_workitem_phase_index = pmanager.check_new_workitem_index()
                            if new_workitem_phase_index <= pending_workitem_candidate.phase_index:
                                pending_workitem = new_workitem
                            else:
                                pending_workitem = pending_workitem_candidate
                                self.pop_pending_workitem_by_id(resourcetype, pending_workitem.id)
                                self.add_back_new_workitem(new_workitem)
                        else:
                            pending_workitem = new_workitem
                    else:
                        if pending_workitem_candidate != None:
                            pending_workitem = pending_workitem_candidate
                            self.pop_pending_workitem_by_id(resourcetype, pending_workitem.id)

                    for workitem in pending_workitem_list:
                        self.add_back_workitem(resourcetype, workitem)
                else:
                    #print('pending_workitems () 2', len (self.get_pending_workitems(resourcetype)))
                    pending_workitem = self.pop_pending_workitem(resourcetype)

                if pending_workitem != None:
                    # pending_workitem.print_data ()
                    if pending_workitem.phase_index == -1:
                        pending_workitem.set_resource_id(resource_id)
                        resource.add_workitem(pending_workitem, resourcetype)
                        pmanager.add_workitem_queue(pending_workitem, self.env.now)
                        pmanager.add_executor(pending_workitem, resource, self.env.now)
                        # new_workitem.print_data ()
                        item_added = True
                    else:
                        next_workitem = pending_workitem.compose_next_workitem(pmanager, resource_id, resourcetype)
                        if next_workitem != None:
                            resource.add_workitem(next_workitem, resourcetype)
                            pmanager.add_executor(next_workitem, resource, self.env.now)
                            # next_workitem.print_data()
                            item_added = True

                #print('pending_workitems () 3', self.get_pending_workitems(resourcetype))

            '''
            if item_added == False:
                #new_workitem = self.create_workitem (imanager, pmanager, resource_id, resourcetype)
                new_workitem = self.get_new_workitem(resourcetype)

                if new_workitem != None:
                    new_workitem.set_resource_id (resource_id)
                    resource.add_workitem (new_workitem, resourcetype)
                    pmanager.add_workitem_queue(new_workitem, self.env.now)
                    pmanager.add_executor (new_workitem, resource, self.env.now)
                    #new_workitem.print_data ()
                    item_added = True
            '''
            if item_added == False:
                #print ('add_workitems ()', resource_id, 'workitems not available')
                break

    def add_new_workitems_old_2 (self, rmanager, imanager, pmanager, empty_resources, resourcetype):
        # print ('add_new_workitems ():')
        completion_times = {}

        for resource in empty_resources:
            completion_time = resource.get_last_completion_time(resourcetype)

            if completion_time == None:
                completion_times[resource.id] = self.env.now
            else:
                completion_times[resource.id] = completion_time

        # print (completion_times)

        sorted_completion_times = dict(sorted(completion_times.items(), key=lambda item: item[1]))

        self.sort_complete_workitems_by_phase_and_stage(resourcetype)

        for resource_id in sorted_completion_times.keys():
            item_added = False
            resource = rmanager.get_resource(resource_id)

            new_workitem = self.get_new_workitem(resourcetype)

            if new_workitem != None:#make sure we process the stage 0's items first since they are not in the pending queue
                index = pmanager.check_new_workitem_index ()
                pending_workitems = self.get_pending_workitems (resourcetype)
                if len (pending_workitems) > 0 and index <= pending_workitems[0].phase_index:
                    new_workitem.set_resource_id(resource_id)
                    resource.add_workitem(new_workitem, resourcetype)
                    pmanager.add_workitem_queue(new_workitem, self.env.now)
                    pmanager.add_executor(new_workitem, resource, self.env.now)
                    # new_workitem.print_data ()
                    item_added = True
                elif len (pending_workitems) <= 0:
                    new_workitem.set_resource_id(resource_id)
                    resource.add_workitem(new_workitem, resourcetype)
                    pmanager.add_workitem_queue(new_workitem, self.env.now, index)
                    pmanager.add_executor(new_workitem, resource, self.env.now)
                    item_added = True
                else:
                    self.add_back_new_workitem (new_workitem)

            if item_added == False:
                pending_workitem = self.pop_pending_workitem(resourcetype)

                if pending_workitem != None:
                    # pending_workitem.print_data ()
                    next_workitem = pending_workitem.compose_next_workitem(pmanager, resource_id, resourcetype)
                    if next_workitem != None:
                        resource.add_workitem(next_workitem, resourcetype)
                        pmanager.add_executor(next_workitem, resource, self.env.now)
                        # next_workitem.print_data()
                        item_added = True

            '''
            if item_added == False:
                #new_workitem = self.create_workitem (imanager, pmanager, resource_id, resourcetype)
                new_workitem = self.get_new_workitem(resourcetype)

                if new_workitem != None:
                    new_workitem.set_resource_id (resource_id)
                    resource.add_workitem (new_workitem, resourcetype)
                    pmanager.add_workitem_queue(new_workitem, self.env.now)
                    pmanager.add_executor (new_workitem, resource, self.env.now)
                    #new_workitem.print_data ()
                    item_added = True
            '''
            if item_added == False:
                #print ('add_workitems ()', resource_id, 'workitems not available')
                break

    def add_new_workitems_DFS_pipelinestage (self, rmanager, imanager, pmanager, empty_resources, resourcetype, pipelinestageindex):
        #print('add_new_workitems_DFS_pipelinestage', pipelinestageindex, len(self.cpuqueue), len(self.gpuqueue))
        completion_times = {}

        for resource in empty_resources:
            completion_time = resource.get_last_completion_time(resourcetype)

            if completion_time == None:
                completion_times[resource.id] = self.env.now
            else:
                completion_times[resource.id] = completion_time

        # print (completion_times)

        sorted_completion_times = dict(sorted(completion_times.items(), key=lambda item: item[1]))

        # print (sorted_completion_times)

        #self.sort_complete_workitems_by_priority(resourcetype)
        # self.sort_complete_workitems_by_stage (resourcetype)
        # self.sort_complete_workitems_by_earliest_schedule_time (resourcetype)
        self.sort_complete_workitems_by_earliest_finish_time (resourcetype)
        # self.sort_complete_workitems_by_latest_finish_time (resourcetype)

        for resource_id in sorted_completion_times.keys():
            # print (resource_id, resourcetype)
            item_added = False
            resource = rmanager.get_resource(resource_id, active=True)

            resubmit_workitem = self.pop_resubmit_workitem(resourcetype)

            if resubmit_workitem != None:
                resubmit_workitem.print_data()
                resubmit_workitem.set_resource_id(resource_id)
                resource.add_workitem(resubmit_workitem, resourcetype)
                pmanager.add_executor(resubmit_workitem, resource.id, self.env.now)
                item_added = True

            if item_added == False:
                pending_workitem = self.pop_pending_workitem_indexed(resourcetype, pipelinestageindex)

                if pending_workitem != None:
                    #print('pending_workitem ()', pending_workitem.id, pending_workitem.version)
                    pending_workitem.set_resource_id(resource_id)
                    resource.add_workitem(pending_workitem, resourcetype)
                    pmanager.add_executor(pending_workitem, resource.id, self.env.now)
                    item_added = True

            if item_added == False:
                # new_workitem = self.create_workitem (imanager, pmanager, resource_id, resourcetype)
                new_workitem = self.get_new_workitem(resourcetype)

                if new_workitem != None:
                    new_workitem.set_resource_id(resource_id)
                    resource.add_workitem(new_workitem, resourcetype)
                    pmanager.add_workitem_queue(new_workitem, self.env.now)
                    pmanager.add_executor(new_workitem, resource.id, self.env.now)
                    # new_workitem.print_data ()
                    item_added = True

            if item_added == False:
                # print ('add_workitems ()', resource_id, 'workitems not available')
                break

        #print ('add_new_workitems_DFS_pipelinestage',pipelinestageindex, len (self.cpuqueue), len (self.gpuqueue))


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
