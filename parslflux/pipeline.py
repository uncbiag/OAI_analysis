import yaml
import sys
from parslfluxsim.resources_sim import Resource
import statistics
import copy
import math

class Phase:
    def __init__(self, pipelinestage, index, resourcetype, rmanager, target):
        self.pipelinestage = pipelinestage
        self.rmanager = rmanager
        self.target = target
        self.resourcetype = resourcetype
        self.active = False
        self.complete = False
        self.index = index
        self.starttime = -1
        self.endtime = -1
        self.current_count = 0
        self.total_count = 0
        self.total_complete = 0
        self.first_resource_release_time = -1
        self.first_workitem_completion_time = -1
        self.add_timestamps = {}
        self.remove_timestamps = {}
        self.end_times_dict = {}
        self.current_executors = []
        self.outputs = []
        self.persistent_outputs = []
        self.workitems = []
        self.pstarttime = -1
        self.pendtime = -1
        self.ptotal_complete = 0
        self.pcurrent_count = 0
        self.ptotal_count = 0
        self.pqueued = {}
        self.pcurrent_executors = []
        self.pfirst_resource_release_time = -1
        self.pfirst_workitem_completion_time = -1
        self.predictions = []
        self.p_outputs = []
        self.persistent_p_outputs = []

    def prediction_reset (self):
        #print ('prediction_reset ()', self.pipelinestage, self.index)
        self.pstarttime = self.starttime
        self.pendtime = self.endtime
        self.ptotal_complete = self.total_complete
        self.pcurrent_count = self.current_count
        self.ptotal_count = self.total_count
        self.pfirst_resource_release_time = self.first_resource_release_time
        self.pfirst_workitem_completion_time = self.first_workitem_completion_time
        self.p_outputs = self.outputs.copy()
        self.persistent_p_outputs = self.persistent_outputs.copy()
        self.pcurrent_executors = self.current_executors.copy()
        self.pqueued = {}

    def get_completed_count (self):
        return self.total_count - self.current_count

    def get_executors (self):
        return self.current_executors

    def add_executor (self, resource, now):
        if resource.id not in self.current_executors:
            self.current_executors.append(resource.id)
            self.pcurrent_executors.append(resource.id)
            if self.total_complete == 0 and self.starttime == -1:
                self.starttime = now
                self.pstarttime = now

    def remove_executor (self, resource, now):
        if resource.id in self.current_executors:
            self.current_executors.remove(resource.id)
            self.pcurrent_executors.remove(resource.id)
            self.end_times_dict[resource.id] = now

    def add_workitem (self, workitem, currenttime):
        if self.active == False:
            self.active = True
            #print(self.pipelinestage, 'activate phase')
        self.current_count += 1
        self.pcurrent_count = self.current_count
        self.total_count += 1
        self.ptotal_count = self.total_count
        self.add_timestamps[workitem.id + ':' + str (currenttime)] = self.current_count
        self.workitems.append(workitem.id)
        #print(self.pipelinestage, 'add workitem', self.index, currenttime, workitem.id, self.current_count, self.total_count)

    def remove_workitem (self, currenttime, workitem):
        self.current_count -= 1
        self.pcurrent_count = self.current_count
        self.remove_timestamps[workitem.id + ':' + str (currenttime)] = self.current_count
        self.workitems.remove (workitem.id)
        self.total_complete += 1
        self.ptotal_complete = self.total_complete
        if self.total_complete == 1:
            self.first_workitem_completion_time = workitem.endtime
            self.pfirst_workitem_completion_time = workitem.endtime

        pipelinestage_resources = self.rmanager.get_resources_type(self.resourcetype)
        if self.total_complete == self.target - len (pipelinestage_resources) + 1:
            self.first_resource_release_time = workitem.endtime
            self.pfirst_resource_release_time = workitem.endtime
        #print(self.pipelinestage, 'remove workitem', self.index, currenttime, workitem.id, self.current_count, self.total_count)

    def close_phase (self):
        self.active = False
        self.complete = True

        timestamps_list = list (self.remove_timestamps.keys())
        '''
        if len (timestamps_list) < 5:
            first_resource_release_timestamp = timestamps_list[0]
        else:
            first_resource_release_timestamp = timestamps_list[-5]
        self.first_resource_release_time = first_resource_release_timestamp.split (':')[1]
        self.pfirst_resource_release_time = first_resource_release_timestamp.split (':')[1]
        '''
        self.endtime = timestamps_list[-1].split (':')[1]
        self.end_times = sorted(self.end_times_dict.items(), key=lambda kv: kv[1])

        print('phase closed', self.pipelinestage.split(':')[0], self.starttime, self.first_workitem_completion_time,
              self.first_resource_release_time, self.endtime, self.total_count, float(self.endtime) - float(self.starttime),
              float(self.endtime) - float(self.first_resource_release_time))

    def get_queued_work (self, rmanager, resourcetype, current_time):
        queued_work = 0
        # work_done, [start_time, fractional_finish_time, whole_finish_time]
        print ('get_queued_work ()', self.pqueued)
        for key in self.pqueued:
            queued_work += (1 - self.pqueued[key][0])
        self.pqueued_work = queued_work

        if self.pqueued_work <= 0.0:
            queued_work = 0
            for executor in self.current_executors:
                executor = rmanager.get_resource(executor)
                work_left = executor.get_work_left(resourcetype, current_time)
                if work_left == None:
                    print('workitem doesnt exist')
                    continue

                queued_work += work_left

            self.queued_work = queued_work

            return self.queued_work
        else:
            return self.pqueued_work

    def print_prediction_data (self):
        print ('print_prediction_data ()', self.pipelinestage)
        print ('print_prediction_data ()', self.pstarttime, self.pendtime, self.ptotal_complete, self.pcurrent_count, self.ptotal_count, self.pfirst_resource_release_time, self.pfirst_workitem_completion_time)
        print ('print_prediction_data ()', self.pqueued)
        print ('print_prediction_data ()', self.pcurrent_executors)

    def print_data (self):
        print('print_data ()', self.pipelinestage, self.starttime, self.first_workitem_completion_time,
              self.first_resource_release_time, self.endtime, self.total_count,
              float(self.endtime) - float(self.starttime),
              float(self.endtime) - float(self.first_resource_release_time),
              len(self.predictions)
              )
        '''
        if len (self.predictions) > 0:
            pexec_time = (float (self.predictions[0][5]) - float(self.predictions[0][4]))
            exec_time = (float(self.endtime) - float(self.starttime))
            print (self.pipelinestage,
                   self.predictions[0][4], self.starttime,
                   self.predictions[0][5], self.endtime)
        [current_time, index, pipelinestage.name.split(':')[0],
         phase.starttime, phase.pstarttime, phase.pendtime,
         phase.pending_output, phase.first_output_completion_time,
         phase.first_resource_release_time])
        print ('prediction ()', self.pipelinestage, )
        '''
        #print (self.timestamps)


class PipelineStage:
    def __init__ (self, stageindex, names, resourcetype, rmanager, target):
        index = 0
        self.name = names[index]
        index += 1
        while index < len (names):
            self.name = self.name + ":" + names[index]
            index += 1
        self.index = stageindex
        self.resourcetype = resourcetype
        self.phases = []
        self.rmanager = rmanager
        self.batchsize = target

    def create_phase (self):
        print(self.name, 'create phase')

        new_phase = Phase(self.name.split(':')[0], len (self.phases), self.resourcetype, self.rmanager, self.batchsize)
        self.phases.append(new_phase)
        return new_phase

    def get_resourcetype (self):
        return self.resourcetype

    def get_index (self):
        return self.index

    def get_name (self):
        return self.name

    def get_phase_length (self):
        index = 0
        for phase in self.phases:
            if phase.active == True or phase.complete == True:
                index += 1
            else:
                break
        return index

    def get_phase (self, workitem):
        return self.phases[workitem.phase_index], workitem.phase_index

    def get_phase_index (self, index):
        #print('get_phase_index', len(self.phases), index)
        if index <= (len(self.phases) - 1):
            return self.phases[index]

        return None

    def add_workitem_index (self, index, workitem, current_time):

        phase = self.phases[index]

        phase.add_workitem (workitem, current_time)

        #print(self.name, 'add workitem index', workitem.id, len(self.phases) - 1, phase.current_count)

    def add_new_workitem (self, workitem, current_time, last_first_phase_closed_index):
        latest_phase = self.phases[last_first_phase_closed_index + 1]

        latest_phase.add_workitem(workitem, current_time)
        workitem.phase_index = last_first_phase_closed_index + 1
        #print (self.name, 'add new workitem', workitem.id, workitem.phase_index, latest_phase.current_count)

    def add_executor (self, workitem, resource, now):
        current_phase, index = self.get_phase(workitem)
        if current_phase == None:
            print ('add_executor', workitem.id, 'not found')
            return
        current_phase.add_executor (resource, now)
        #print(self.name, 'add executor', now, workitem.id, len(self.phases) - 1, index, self.phases[index].total_complete)

    def remove_executor (self, workitem, resource, now):
        current_phase, index = self.get_phase(workitem)
        if current_phase == None:
            print('remove_executor', workitem.id, self.name, 'not found')
            return
        current_phase.remove_executor (resource, now)
        #print(self.name, 'remove executor', workitem.id, len(self.phases) - 1, index)

    def add_output (self, workitem):
        phase, index = self.get_phase(workitem)
        phase.outputs.append(workitem.endtime)
        phase.persistent_outputs.append (workitem.endtime)
        phase.p_outputs.append(workitem.endtime)
        phase.persistent_p_outputs.append(workitem.endtime)

    def remove_output (self, workitem):
        phase, index = self.get_phase(workitem)
        phase.outputs.pop (0)
        phase.p_outputs.pop (0)

    def get_resource_service_rate (self, rmanager):
        pipelinestage_resources = rmanager.get_resources_type(self.resourcetype)

        #print ('get_resource_service_rate ()', self.name)
        all_resource_service_rates = []
        for resource in pipelinestage_resources:
            exectime = resource.get_exectime(self.name)
            if exectime == 0:
                print('exectime does not exist')
                continue
            all_resource_service_rates.append(1 / exectime)

        self.all_resource_service_rate = sum(all_resource_service_rates)

        #print('get_resource_service_rate ()', self.all_resource_service_rate)

        return self.all_resource_service_rate

    def get_avg_resource_service_rate (self, rmanager):
        #print ('get_avg_resource_service_rate ()', self.name)
        pipelinestage_resources = rmanager.get_resources_type(self.resourcetype)

        all_resource_service_rates = []
        for resource in pipelinestage_resources:
            exectime = resource.get_exectime(self.name)
            if exectime == 0:
                print('exectime does not exist')
                continue
            all_resource_service_rates.append(1 / exectime)

        if len (all_resource_service_rates) <= 0:
            return 0

        self.avg_resource_service_rate = sum(all_resource_service_rates) / len (all_resource_service_rates)

        #print ('get_avg_resource_service_rate ()', self.avg_resource_service_rate)

        return self.avg_resource_service_rate

    def get_time_required_2 (self, rmanager, target, current_time, phase_index):
        print ('get_time_required_2 ()', self.name, target, current_time)

        pipelinestage_resources = rmanager.get_resources_type(self.resourcetype)

        phase = self.phases[phase_index]

        resource_data = {}
        times_taken = {}

        total_queued_work = 0

        #print ('get_time_required_2 ()', phase.pcurrent_executors)
        #print ('get_time_required_2 ()', phase.pqueued)

        for resource_id in phase.pcurrent_executors:
            resource = rmanager.get_resource (resource_id)
            queued_work = resource.get_work_left(self.resourcetype, current_time)
            #print (resource_id, queued_work)
            if (queued_work == None or queued_work == 0) and resource_id in phase.pqueued.keys ():
                queued_work = 1 - phase.pqueued[resource_id][0]

            exectime = resource.get_exectime(self.name)
            service_rate = 1 / exectime
            times_required = queued_work / service_rate
            resource_data[resource.id] = [queued_work, times_required, service_rate, resource.id]
            total_queued_work += queued_work

        #print ('get_time_required_2 ()', total_queued_work)

        resource_index = 0
        while resource_index < len (pipelinestage_resources):
            resource = pipelinestage_resources[resource_index]
            if resource.id in phase.pcurrent_executors and resource_data[resource.id][0] > 0.0:
                resource_index += 1
                continue

            if total_queued_work + 1 > target:
                resource_data[resource.id] = [0, 0, 0, resource.id]
                times_taken[resource.id] = 0.0
            else:
                queued_work = 1
                exectime = resource.get_exectime(self.name)
                service_rate = 1 / exectime
                times_required = queued_work / service_rate
                resource_data[resource.id] = [queued_work, times_required, service_rate, resource.id]
                total_queued_work += queued_work

            resource_index += 1

        #print ('get_time_required_2 ()', resource_data)
        #print ('get_time_required_2 ()', times_taken)

        while True:
            sorted_resource_items = sorted(resource_data.items(), key=lambda kv: kv[1][1])
            selected_resource_key = None
            for resource in sorted_resource_items:
                time_taken = resource[1][1]
                if time_taken > 0.0:
                    selected_resource_key = resource[0]
                    break
            if time_taken <= 0.0:
                break
            #print (target, total_queued_work, time_taken, sorted_resource_items)
            #print (times_taken)

            for resource_key in resource_data.keys ():
                resource = resource_data[resource_key]
                if resource[0] <= 0.0:
                    continue
                if resource_key == selected_resource_key:
                    resource[0] = 0.0
                else:
                    work_done = resource[2] * time_taken
                    resource[0] -= work_done
                if resource_key not in times_taken.keys ():
                    times_taken[resource_key] = time_taken
                else:
                    times_taken[resource_key] += time_taken
                #print (resource)
                if resource[0] <= 0.0:
                    if total_queued_work + 1 <= target:
                        resource[0] = 1
                        total_queued_work += 1
                resource[1] = resource[0] / resource[2]

        sorted_times = sorted(times_taken.items(), key=lambda kv: kv[1])

        end_times = []
        for end_time in sorted_times:
            new_end_time = list (end_time)
            end_times.append(new_end_time)
        print ('get_time_required_2 ()', end_times)

        return end_times[-1][1], end_times[0][1], end_times

    def get_possible_work_done (self, rmanager, idle_times, target, fractional_work_allowed):
        execution_times = {}
        fractional_work = {}
        output_times = []
        work_done = 0
        assigned_work = 0

        #print ('get_possible_work_done () idle_times', idle_times[2].items ())

        sorted_idle_times = dict(sorted(idle_times[2].items(), key=lambda kv: kv[1][-1][0]))
        starttime = None

        while assigned_work < target:
            #print('get_possible_work_done ():', work_done, sorted_idle_times)
            assigned_work_snapshot = assigned_work

            resource_index = 0
            fractional_work_done = False
            while resource_index < len (list (sorted_idle_times.keys ())):
                resource_key = list (sorted_idle_times.keys ())[resource_index]
                resource_data = sorted_idle_times[resource_key][-1]

                time_left = resource_data[2]

                resource = rmanager.get_resource(resource_key)
                resource_service_rate = 1 / resource.get_exectime(self.name)
                time_taken = 1 / resource_service_rate

                #print ('get_possible_work_done ()', time_taken, time_left, resource.id)

                if time_left >= time_taken:
                    work_done += 1
                    assigned_work += 1
                    expired = time_taken

                    if starttime == None:
                        starttime = resource_data[0]

                    if resource.id not in execution_times:
                        execution_times[resource.id] = [resource_data[0], resource_data[0] + expired, work_done]
                        output_times.append (resource_data[0] + time_taken)
                    else:
                        output_times.append(execution_times[resource.id][1] + time_taken)
                        execution_times[resource.id][1] += expired
                        execution_times[resource.id][2] += work_done

                    resource_data[2] -= expired
                    resource_data[0] += expired
                    break
                elif time_left > 0.0 and fractional_work_allowed == True:
                    assigned_work += 1
                    if starttime == None:
                        starttime = resource_data[0]
                    #work_done, [start_time, fractional_finish_time, whole_finish_time]
                    fractional_work[resource_key] = [1 * (time_left / time_taken), [resource_data[0], resource_data[1], resource_data[0] + time_taken]]
                    expired = time_left
                    resource_data[2] -= expired
                    resource_data[0] += expired
                    fractional_work_done = True
                    break

                resource_index += 1

            sorted_idle_times = dict(sorted(idle_times[2].items(), key=lambda kv: kv[1][-1][0]))

            if assigned_work_snapshot == assigned_work and fractional_work_done == False:
                break

        idle_times[0] = list (sorted_idle_times.values())[0][-1][0]
        idle_times[1] = list (sorted_idle_times.values())[0][-1][1]
        idle_times[2] = sorted_idle_times

        output_times.sort()

        print ('------------------------------------')
        print ('remaining idle times ()', idle_times)
        print ('work done ()', work_done)
        print ('output times ()', output_times)
        print ('fraction work ()', fractional_work)
        print('------------------------------------')

        return starttime, work_done, idle_times, output_times, fractional_work

    def get_possible_work_done_2 (self, rmanager, idle_times,
                                  target, prev_output_times, fractional_work_allowed):
        fractional_work = {}
        output_times = []
        work_done = 0

        starttime = None

        #print ('get_possible_work_done_2 () idle_times', idle_times[2].items())
        #print ('get_possible_work_done_2 () prev_output_times', prev_output_times)

        sorted_idle_times = dict (sorted(idle_times[2].items(), key=lambda kv: kv[1][-1][0]))

        while work_done < target:
            #print('get_possible_work_done_2 ():', work_done, sorted_idle_times)
            work_done_snapshot = work_done

            fractional_work_done = False
            resource_index = 0
            while resource_index < len(list(sorted_idle_times.keys())) and len (prev_output_times) > 0:

                resource_key = list(sorted_idle_times.keys())[resource_index]
                resource_data = sorted_idle_times[resource_key][-1]

                #print ('get_possible_work_done_2 ()', 'resource_data', resource_key, resource_data)
                #print ('get_possible_work_done_2 ()', 'prev_output', prev_output_times)

                if prev_output_times[0] >= resource_data[1]:
                    #print ('get_possible_work_done_2 ()', prev_output_times[0], resource_data[1])
                    resource_index += 1
                    continue

                time_left = resource_data[1] - prev_output_times[0]

                if time_left >= resource_data[2]:
                    time_left = resource_data[2]
                    start_time = resource_data[0]
                else:
                    start_time = prev_output_times[0]

                resource = rmanager.get_resource(resource_key)
                resource_service_rate = 1 / resource.get_exectime(self.name)
                time_taken = 1 / resource_service_rate

                #print ('get_possible_work_done_2 ()', time_taken, time_left, resource.id, start_time, prev_output_times[0])

                if time_left >= time_taken:
                    if starttime == None:
                        starttime = start_time
                    work_done += 1
                    expired = time_taken
                    output_times.append(start_time + time_taken)

                    if start_time > resource_data[0]:
                        #split idle times
                        sorted_idle_times[resource_key].insert (-1, [resource_data[0], start_time, start_time - resource_data[0]])

                    resource_data[0] = (start_time + expired)
                    resource_data[2] = resource_data[1] - resource_data[0]
                    prev_output_times.pop (0)
                    break
                elif time_left > 0.0 and fractional_work_allowed == True:
                    if starttime == None:
                        starttime = start_time

                    if start_time > resource_data[0]:
                        #split idle times
                        sorted_idle_times[resource_key].insert (-1, [resource_data[0], start_time, start_time - resource_data[0]])

                        # work_done, [start_time, fractional_finish_time, whole_finish_time]
                    fractional_work[resource_key] = [1 * (time_left / time_taken),
                                                     [start_time, resource_data[1],
                                                      start_time + time_taken]]

                    expired = time_left
                    resource_data[0] = (start_time + expired)
                    resource_data[2] = resource_data[1] - resource_data[0]

                    fractional_work_done = True
                    prev_output_times.pop (0)
                    break

                resource_index += 1

            sorted_idle_times = dict(sorted(idle_times[2].items(), key=lambda kv: kv[1][-1][0]))

            if work_done_snapshot == work_done and fractional_work_done == False:
                break

        idle_times[0] = list(sorted_idle_times.values())[0][-1][0]
        idle_times[1] = list(sorted_idle_times.values())[0][-1][1]
        idle_times[2] = sorted_idle_times

        output_times.sort()

        print('------------------------------------')
        print('remaining idle times ()', idle_times)
        print('work done ()', work_done)
        print('output times ()', output_times)
        print('fraction work ()', fractional_work)
        print('------------------------------------')

        return starttime, work_done, idle_times, output_times, prev_output_times, fractional_work


    def print_data (self, index):
        if index < len (self.phases):
            self.phases[index].print_data ()

class PipelineManager:
    def __init__ (self, pipelinefile, budget, batchsize, max_images):
        self.pipelinefile = pipelinefile
        self.pipelinestages = []
        self.batchsize = batchsize
        self.budget = budget
        self.last_last_phase_closed_index = -1
        self.last_first_phase_closed_index = -1
        self.no_of_columns = int(math.ceil(max_images / batchsize))

    def parse_pipelines (self, rmanager):
        pipelinedatafile = open (self.pipelinefile)
        pipelinedata = yaml.load (pipelinedatafile, Loader = yaml.FullLoader)

        #parse pipeline stages
        pipelinestage_names = []
        pipelinestage_resourcetypes = []

        for pipelinestage in pipelinedata['pipelinestages']:
            pipelinestage_names.append (pipelinestage['name'])
            pipelinestage_resourcetypes.append (pipelinestage['resourcetype'])

        index = 0
        current_resourcetype = pipelinestage_resourcetypes[index]
        current_names = []
        current_names.append(pipelinestage_names[index])
        current_index = index

        index += 1
        while index < len (pipelinestage_resourcetypes):
            if current_resourcetype == pipelinestage_resourcetypes[index]:
                current_names.append(pipelinestage_names[index])
            else:
                self.pipelinestages.append (PipelineStage(current_index, current_names, current_resourcetype, rmanager, self.batchsize))
                current_index = index
                current_names.clear()
                current_resourcetype = pipelinestage_resourcetypes[index]
                current_names.append(pipelinestage_names[index])
            index += 1
        self.pipelinestages.append(PipelineStage(current_index, current_names, current_resourcetype, rmanager, self.batchsize))

    def get_idle_periods(self, end_times, finish_time):
        idle_periods = {}
        for end_time in end_times:
            resource_id = end_time[0]
            resource_finish_time = end_time[1]

            idle_periods[resource_id] = [[resource_finish_time, finish_time, finish_time - resource_finish_time]]
            #idle_periods[resource_id] = finish_time - resource_finish_time

        idle_periods = dict(sorted(idle_periods.items(), key=lambda item: item[1][-1][2], reverse=True))

        return idle_periods

    def prediction_reset (self, start_index):
        for pipelinestage in self.pipelinestages:
            for current_index in range(start_index, len (pipelinestage.phases)):
                pipelinestage.phases[current_index].prediction_reset ()

    def predict_execution_fixed (self, rmanager, current_time, batchsize, last_phase_index_closed, no_of_prediction_phases):
        index = last_phase_index_closed + 1

        total_cpu_idle_periods = {}
        total_gpu_idle_periods = {}

        print(index, '###############################################################')

        self.prediction_reset (index)

        while index < last_phase_index_closed + (no_of_prediction_phases + 1) and index < self.no_of_columns:

            cpu_idle_periods = []
            gpu_idle_periods = []

            prev_prev_pipelinestage_phase = None
            prev_pipelinestage_phase = None
            prev_pipelinestage = None
            prev_prev_pipelinestage = None

            pipelinestage_index = 0
            for pipelinestage in self.pipelinestages:

                phase = pipelinestage.phases[index]

                if phase.active == False and phase.complete == True: #not neeeded
                    pipelinestage_index += 1
                    continue

                # Calculate queued work
                queued_work = phase.get_queued_work(rmanager, pipelinestage.resourcetype, current_time)

                # calculate service rates
                all_resource_service_rate = pipelinestage.get_resource_service_rate(rmanager)
                avg_resource_service_rate = pipelinestage.get_avg_resource_service_rate(rmanager)
                pipelinestage_resources = rmanager.get_resources_type(pipelinestage.resourcetype)

                phase.target = batchsize
                phase.pending_output = phase.target - phase.ptotal_complete
                work_to_be_done = queued_work + phase.pcurrent_count - len(phase.pcurrent_executors) + (phase.target - phase.ptotal_count)

                if phase.pending_output == 0:
                    pipelinestage_index += 1
                    continue

                print(phase.pipelinestage, queued_work, phase.current_count, phase.pcurrent_count, phase.total_complete, phase.ptotal_complete, phase.total_count, phase.ptotal_count)

                if prev_pipelinestage_phase == None:

                    if phase.pstarttime == -1:
                        phase.pstarttime = current_time
                        print ('setting pstarttime')
                    if phase.pfirst_workitem_completion_time == -1:
                        phase.pfirst_workitem_completion_time = phase.pstarttime + 1 / avg_resource_service_rate  # use fastest
                        print ('setting pfirst_workitem_completion_time')

                    finish_time, first_resource_release_time, end_times = pipelinestage.get_time_required_2(rmanager, work_to_be_done, current_time, index)

                    phase.pendtime = current_time + finish_time

                    for end_time in end_times:
                        end_time[1] = current_time + end_time[1]
                    phase.end_times = end_times

                    if phase.pfirst_resource_release_time == -1:
                        phase.pfirst_resource_release_target = batchsize - (len (pipelinestage_resources) - 1)
                        phase.pfirst_resource_release_time = current_time + first_resource_release_time
                        print ('setting pfirst_resource_release_time')

                    phase.predictions.append(
                        [current_time, index, pipelinestage.name.split(':')[0],
                         phase.starttime, phase.pstarttime, phase.pendtime,
                         phase.pending_output, phase.pfirst_workitem_completion_time, phase.pfirst_resource_release_time])
                else:
                    prev_sametype_phase = prev_prev_pipelinestage_phase

                    if phase.pstarttime == -1:
                        print ('setting pstarttime', current_time, prev_pipelinestage_phase.pfirst_workitem_completion_time)
                        if prev_sametype_phase == None:
                            if current_time > prev_pipelinestage_phase.pfirst_workitem_completion_time:
                                phase.pstarttime = current_time
                            else:
                                phase.pstarttime = prev_pipelinestage_phase.pfirst_workitem_completion_time
                        else:
                            print ('setting pstarttime', prev_sametype_phase.pfirst_resource_release_time)
                            if prev_pipelinestage_phase.pfirst_workitem_completion_time > prev_sametype_phase.pfirst_resource_release_time:
                                phase.pstarttime = prev_pipelinestage_phase.pfirst_workitem_completion_time
                            else:
                                phase.pstarttime = prev_sametype_phase.pfirst_resource_release_time

                            if current_time > phase.pstarttime:
                                phase.pstarttime = current_time

                    if phase.pfirst_workitem_completion_time == -1:
                        print ('setting pfirst_workitem_completion_time')
                        phase.pfirst_workitem_completion_time = phase.pstarttime + 1 / avg_resource_service_rate

                    if prev_sametype_phase != None:
                        if prev_pipelinestage_phase.pfirst_workitem_completion_time > prev_sametype_phase.pfirst_resource_release_time:
                            idle_periods = self.get_idle_periods(prev_sametype_phase.end_times, phase.pstarttime)
                            idle_period_start = list(idle_periods.values())[0][-1][0]

                            print('idle period 1', phase.pipelinestage, [idle_period_start, phase.pstarttime],
                                  idle_periods)

                            if phase.resourcetype == 'CPU':
                                cpu_idle_periods.append([idle_period_start, phase.pstarttime, idle_periods])
                            else:
                                gpu_idle_periods.append([idle_period_start, phase.pstarttime, idle_periods])

                    if prev_pipelinestage.all_resource_service_rate > pipelinestage.all_resource_service_rate:
                        finish_time, first_resource_release_time, end_times = pipelinestage.get_time_required_2(rmanager,
                                                                                                     work_to_be_done,
                                                                                                     current_time,
                                                                                                     index)
                        '''
                        if phase.starttime == -1:
                            phase.pendtime = phase.pstarttime + finish_time
                        else:
                            phase.pendtime = current_time + finish_time
                            
                        for end_time in end_times:
                            if phase.starttime == -1:
                                end_time[1] = phase.pstarttime + end_time[1]
                            else:
                                end_time[1] = current_time + end_time[1]
                        '''
                        if phase.pstarttime < current_time:
                            phase.pendtime = current_time + finish_time
                        else:
                            phase.pendtime = phase.pstarttime + finish_time

                        for end_time in end_times:
                            if phase.pstarttime < current_time:
                                end_time[1] = current_time + end_time[1]
                            else:
                                end_time[1] = phase.pstarttime + end_time[1]

                        phase.end_times = end_times

                        if phase.pfirst_resource_release_time == -1:
                            print ('setting pfirst_resource_release_time')
                            phase.pfirst_resource_release_target = batchsize - (len(pipelinestage_resources) - 1)
                            if phase.pstarttime < current_time:
                                phase.pfirst_resource_release_time = current_time + first_resource_release_time
                            else:
                                phase.pfirst_resource_release_time = phase.pstarttime + first_resource_release_time

                        if pipelinestage_index == len(self.pipelinestages) - 1:
                            idle_periods = self.get_idle_periods (phase.end_times, phase.pendtime)

                            print('idle period 2', phase.pipelinestage,
                                  [end_times[0][1], end_times[-1][1]],
                                  idle_periods)

                            if phase.resourcetype == 'CPU':
                                cpu_idle_periods.append([end_times[0][1], end_times[-1][1], idle_periods])
                            else:
                                gpu_idle_periods.append([end_times[0][1], end_times[-1][1], idle_periods])

                        '''
                        if phase.pfirst_resource_release_time == -1:
                            phase.pfirst_resource_release_target = batchsize - (len(pipelinestage_resources) - 1)

                            #work_done = phase.total_complete + (phase.current_count - queued_work)
                            #work_to_be_done = phase.first_resource_release_target - work_done

                            if phase.starttime == -1:
                                phase.first_resource_release_time = phase.pstarttime + first_resource_release_time
                            else:
                                phase.first_resource_release_time = current_time + first_resource_release_time
                        '''

                        phase.predictions.append(
                            [current_time, index, pipelinestage.name.split(':')[0],
                             phase.starttime, phase.pstarttime, phase.pendtime,
                             phase.pending_output, phase.pfirst_workitem_completion_time,
                             phase.pfirst_resource_release_time])
                    else:
                        queued_finish_time, _, _ = pipelinestage.get_time_required_2(rmanager,
                                                                                     queued_work + phase.pcurrent_count - len (phase.pcurrent_executors),
                                                                                     current_time, index)
                        if phase.pstarttime < current_time:
                            queued_finish_time = current_time + queued_finish_time
                        else:
                            queued_finish_time = phase.pstarttime + queued_finish_time

                        print(phase.pipelinestage, queued_finish_time)

                        finish_time, first_resource_release_time, end_times = pipelinestage.get_time_required_2(
                                                                                rmanager,
                                                                                work_to_be_done,
                                                                                current_time,
                                                                                index)

                        print(phase.pipelinestage, queued_finish_time, finish_time)

                        if queued_finish_time >= prev_pipelinestage_phase.pendtime:
                            if phase.pstarttime < current_time:
                                phase.pendtime = current_time + finish_time
                            else:
                                phase.pendtime = phase.pstarttime + finish_time

                            for end_time in end_times:
                                if phase.pstarttime < current_time:
                                    end_time[1] = current_time + end_time[1]
                                else:
                                    end_time[1] = phase.pstarttime + end_time[1]

                            phase.end_times = end_times

                            if phase.pfirst_resource_release_time == -1:
                                phase.pfirst_resource_release_target = batchsize - (len(pipelinestage_resources) - 1)
                                if phase.pstarttime < current_time:
                                    phase.pfirst_resource_release_time = current_time + first_resource_release_time
                                else:
                                    phase.pfirst_resource_release_time = phase.pstarttime + first_resource_release_time
                        else:
                            phase.pendtime = prev_pipelinestage_phase.pendtime + 1 / avg_resource_service_rate

                            phase.end_times = []

                            for resource in pipelinestage_resources:
                                phase.end_times.append([resource.id, phase.pendtime])

                            if phase.pfirst_resource_release_time == -1:
                                phase.pfirst_resource_release_time = prev_pipelinestage_phase.pendtime + 1 / avg_resource_service_rate

                            if phase.pstarttime < current_time:
                                predicted_time_to_completion = phase.pendtime - current_time
                            else:
                                predicted_time_to_completion = phase.pendtime - phase.pstarttime
                            fastest_time_to_completion = finish_time

                            idle_periods = {}
                            idle_period = predicted_time_to_completion - fastest_time_to_completion

                            for resource in pipelinestage_resources:
                                idle_periods[resource.id] = [[queued_finish_time,
                                                             queued_finish_time + idle_period,
                                                             idle_period]]

                            # TO-DO: change starttime to time after induced idle periods
                            print ('idle period 3', phase.pipelinestage, [queued_finish_time,
                                   queued_finish_time + idle_period],
                                   idle_periods)

                            if phase.resourcetype == 'CPU':
                                cpu_idle_periods.append ([queued_finish_time, queued_finish_time + idle_period, idle_periods])
                            else:
                                gpu_idle_periods.append ([queued_finish_time, queued_finish_time + idle_period, idle_periods])

                        phase.predictions.append(
                            [current_time, index, pipelinestage.name.split(':')[0],
                             phase.starttime, phase.pstarttime, phase.pendtime,
                             phase.pending_output, phase.pfirst_workitem_completion_time,
                             phase.pfirst_resource_release_time])

                    if pipelinestage_index == len (self.pipelinestages) - 1:
                        idle_periods = self.get_idle_periods(prev_pipelinestage_phase.end_times, phase.pendtime)

                        print ('idle period 4', prev_pipelinestage_phase.pipelinestage, [prev_pipelinestage_phase.pendtime, phase.pendtime], idle_periods)

                        if prev_pipelinestage_phase.resourcetype == 'CPU':
                            cpu_idle_periods.append ([prev_pipelinestage_phase.pendtime, phase.pendtime, idle_periods])
                        else:
                            gpu_idle_periods.append ([prev_pipelinestage_phase.pendtime, phase.pendtime, idle_periods])

                prediction = phase.predictions[-1]

                output = ""
                for item in prediction:
                    output += " " + str(item)

                #print ('predict_execution ()', phase.end_times)

                print ('predict_execution ()', output)

                prev_prev_pipelinestage = prev_pipelinestage
                prev_pipelinestage = pipelinestage
                prev_prev_pipelinestage_phase = prev_pipelinestage_phase
                prev_pipelinestage_phase = phase
                pipelinestage_index += 1

            print(index, '###############################################################')
            index += 1

            print('cpu idle periods before precompute ()', index, cpu_idle_periods)
            print('gpu idle periods before precompute ()', index, gpu_idle_periods)

            self.pre_compute_prediction(rmanager, batchsize, cpu_idle_periods, gpu_idle_periods, index)

            print(index - 1, '***************************************************************')

            current_time = phase.pendtime

            #new_cpu_idle_periods, new_gpu_idle_periods, early_outputs_dict, work_done_dict = \
            #    self.calculate_early_computation(rmanager, batchsize, cpu_idle_periods, gpu_idle_periods, index, phase.pendtime)

        self.prediction_reset(last_phase_index_closed + 1)

    def pre_compute_prediction (self, rmanager, batchsize, cpu_idle_periods, gpu_idle_periods, phase_index):

        current_phase_index = phase_index
        new_cpu_idle_periods = cpu_idle_periods
        new_gpu_idle_periods = gpu_idle_periods

        #print('print_prediction_data ()', self.pipelinestage)
        #print('print_prediction_data ()', self.pstarttime, self.pendtime, self.ptotal_complete, self.pcurrent_count,
        #      self.ptotal_count, self.pfirst_resource_release_time, self.pfirst_workitem_completion_time)
        #print('print_prediction_data ()', self.pqueued)
        #print('print_prediction_data ()', self.pcurrent_executors)

        #sorted_cpu_idle_periods, sorted_gpu_idle_periods, starttimes_dict, output_times_dict, p_output_times_dict, work_done_dict, fractional_work_dict
        while current_phase_index < self.no_of_columns:
            if phase_index == current_phase_index:
                fractional_work_allowed = True
            else:
                fractional_work_allowed = False
            new_cpu_idle_periods, new_gpu_idle_periods, starttimes_dict, work_done_dict, fractional_work_dict =\
                self.calculate_early_computation (rmanager, batchsize, new_cpu_idle_periods, new_gpu_idle_periods, current_phase_index, fractional_work_allowed)


            pipelinestage_index = 0
            for pipelinestage_name in work_done_dict.keys ():
                current_pipelinestage = self.pipelinestages[pipelinestage_index]
                current_phase = current_pipelinestage.phases[current_phase_index]

                pipelinestage_resources = rmanager.get_resources_type(current_pipelinestage.resourcetype)

                print('pre_compute_prediction ()', pipelinestage_name, fractional_work_dict[pipelinestage_name])
                print('pre_compute_prediction ()', pipelinestage_name, current_phase.persistent_p_outputs)

                next_phase = None
                if pipelinestage_index < len (self.pipelinestages):
                    next_phase = self.pipelinestages[pipelinestage_index + 1].phases[current_phase_index]

                if current_phase.pstarttime == -1:
                    current_phase.pstarttime = starttimes_dict[pipelinestage_name]

                if work_done_dict[pipelinestage_name] > 0.0:
                    if current_phase.pfirst_workitem_completion_time == -1:
                        current_phase.pfirst_workitem_completion_time = current_phase.persistent_p_outputs[0]

                    current_phase.ptotal_complete += work_done_dict[pipelinestage_name]

                    if current_phase.ptotal_complete > (batchsize - len (pipelinestage_resources)):
                        if current_phase.pfirst_resource_release_time == -1:
                            current_phase.pfirst_resource_release_time = current_phase.persistent_p_outputs[batchsize - len (pipelinestage_resources)]

                    if current_phase.ptotal_complete == batchsize:
                        current_phase.pendtime = current_phase.persistent_p_outputs[-1]

                    if pipelinestage_index > 0:
                        current_phase.pcurrent_count -= work_done_dict[pipelinestage_name]

                    if next_phase != None:
                        next_phase.pcurrent_count += work_done_dict[pipelinestage_name]

                # work_done, start_time, fractional_finish_time, whole_finish_time
                if len (fractional_work_dict[pipelinestage_name].keys ()) > 0:
                    current_phase.pcurrent_executors = []
                    current_phase.pqueued = {}
                    for resource_key in fractional_work_dict[pipelinestage_name]:
                        current_phase.pqueued[resource_key] = fractional_work_dict[pipelinestage_name][resource_key]
                        current_phase.pcurrent_executors.append(resource_key)

                    if pipelinestage_index == 0:
                        current_phase.pcurrent_count = len (fractional_work_dict[pipelinestage_name].keys ())

                current_phase.ptotal_count = (current_phase.ptotal_complete + current_phase.pcurrent_count)

                current_phase.print_prediction_data()
                if next_phase != None:
                    next_phase.print_prediction_data()

                if work_done_dict[pipelinestage_name] <= 0.0:
                    break

                pipelinestage_index += 1

            if pipelinestage_index == 0:
                break

            current_phase_index += 1

        print ('pre_compute_prediction cpu idle periods ()', phase_index, cpu_idle_periods)
        print ('pre_compute_prediction gpu idle periods ()', phase_index, gpu_idle_periods)

    def calculate_early_computation (self, rmanager, batchsize, cpu_idle_periods, gpu_idle_periods, phase_index, first_index):
        from operator import itemgetter
        sorted_cpu_idle_periods = sorted(cpu_idle_periods, key=itemgetter(0))
        sorted_gpu_idle_periods = sorted(gpu_idle_periods, key=itemgetter(0))

        print('sorted cpu idle periods ()', phase_index, sorted_cpu_idle_periods)
        print('sorted gpu idle periods ()', phase_index, sorted_gpu_idle_periods)

        work_done_dict = {}
        fractional_work_dict = {}
        starttimes_dict = {}

        previous_pipelinestage = None

        for pipelinestage in self.pipelinestages:

            current_phase = pipelinestage.phases[phase_index]
            if previous_pipelinestage != None:
                previous_phase = previous_pipelinestage.phases[phase_index]

            if pipelinestage.resourcetype == 'CPU':
                idle_periods = sorted_cpu_idle_periods
            else:
                idle_periods = sorted_gpu_idle_periods

            if previous_pipelinestage == None:
                target = batchsize - (current_phase.ptotal_complete + len (current_phase.pcurrent_executors))
            else:
                target = current_phase.pcurrent_count + work_done_dict[previous_pipelinestage.name]

            work_done_dict[pipelinestage.name] = 0
            fractional_work_dict[pipelinestage.name] = {}
            starttimes_dict[pipelinestage.name] = -1

            if target == 0:
                previous_pipelinestage = pipelinestage
                #print ('calculate_early_computation', 'target is zero')
                continue

            idle_period_index = 0

            while idle_period_index < len (idle_periods) and work_done_dict[pipelinestage.name] < target:
                idle_period = idle_periods[idle_period_index]

                print ('calculate_early_computation', pipelinestage.name, idle_period_index, target, idle_period)

                if previous_pipelinestage == None:
                    qualified = True
                else:
                    #output_times = output_times_dict[previous_pipelinestage.name]
                    output_times = previous_phase.p_outputs
                    if len (output_times) <= 0 or idle_period[1] < output_times[0]:
                        qualified = False
                    else:
                        qualified = True

                if previous_pipelinestage != None and qualified == False:
                    idle_period_index += 1
                    continue

                if idle_period_index == len (idle_periods) - 1 and first_index == True:
                    fractional_work_allowed = True
                else:
                    fractional_work_allowed = False

                if previous_pipelinestage == None:
                    starttime, work_done, idle_times, new_output_times, fractional_work = \
                        pipelinestage.get_possible_work_done (rmanager, idle_periods[idle_period_index],
                                                              target - work_done_dict[pipelinestage.name],
                                                              fractional_work_allowed)

                else:
                    starttime, work_done, idle_times, new_output_times, old_output_times, fractional_work = \
                        pipelinestage.get_possible_work_done_2 (rmanager, idle_periods[idle_period_index],
                                                                target - work_done_dict[pipelinestage.name],
                                                                output_times,
                                                                fractional_work_allowed)
                    previous_phase.p_outputs = old_output_times

                if work_done > 0:
                    #print('calculate_early_computation', pipelinestage.name, idle_period_index, work_done)
                    idle_periods[idle_period_index] = idle_times
                    current_phase.p_outputs.extend (new_output_times)
                    current_phase.persistent_p_outputs.extend (new_output_times.copy ())
                    work_done_dict[pipelinestage.name] += work_done

                if starttime != None and starttimes_dict[pipelinestage.name] == -1:
                    starttimes_dict[pipelinestage.name] = starttime

                if fractional_work_allowed == True:
                    fractional_work_dict[pipelinestage.name] = fractional_work

                idle_period_index += 1

            previous_pipelinestage = pipelinestage

        print('new sorted cpu idle periods ()', sorted_cpu_idle_periods)
        print('new sorted gpu idle periods ()', sorted_gpu_idle_periods)

        return sorted_cpu_idle_periods, sorted_gpu_idle_periods, starttimes_dict, work_done_dict, fractional_work_dict


    def close_phases_fixed (self, rmanager, anyway):
        cpu_resources = rmanager.get_resources_type('CPU')
        gpu_resources = rmanager.get_resources_type('GPU')

        index = self.last_last_phase_closed_index + 1

        length = len(self.pipelinestages[0].phases)

        # print ('close phases size', init_length)
        latest_last_phase_closed = None

        while index < length:
            current_index_pipelinestage_index = 0
            for pipelinestage in self.pipelinestages:
                phase = pipelinestage.phases[index]

                if phase.active == False and phase.complete == True:
                    current_index_pipelinestage_index += 1
                    continue
                elif phase.active == False:
                    break

                if phase.total_complete == self.batchsize or anyway == True:
                    phase.close_phase()
                    if current_index_pipelinestage_index == len (self.pipelinestages) - 1:
                        latest_last_phase_closed = index
                        self.last_last_phase_closed_index = index
                    elif current_index_pipelinestage_index == 0:
                        self.last_first_phase_closed_index = index
                current_index_pipelinestage_index += 1

            index += 1

        return latest_last_phase_closed

    def close_phases (self, rmanager, nowtime):
        #print ('close phases')

        cpu_resources = rmanager.get_resources_type ('CPU')
        gpu_resources = rmanager.get_resources_type ('GPU')

        index = 0

        init_length = self.pipelinestages[0].get_phase_length ()

        #print ('close phases size', init_length)

        while index < init_length:
            #print ('close phases index', index)
            prev_index_closed = False

            if index == 0:
                prev_index_closed = True
            else:
                prev_index_last_pipelinestage_phase = self.pipelinestages[-1].phases[index - 1]

                #print ('close_phases last stage size', last_pipelinestage.get_phase_length ())
                #print ('close_phases', index, last_pipelinestage.phases[index - 1].pipelinestage, last_pipelinestage.phases[index - 1].active)
                if prev_index_last_pipelinestage_phase.active == False:
                    prev_index_closed = True

            if prev_index_closed == False:
                break

            current_index_pipelinestage_index = 0

            for pipelinestage in self.pipelinestages:
                phase = pipelinestage.phases[index]

                if phase.active == False and phase.complete == True:
                    current_index_pipelinestage_index += 1
                    continue
                elif phase.active == False:
                    break

                if current_index_pipelinestage_index == 0:
                    prev_pipelinestage_phase_closed = True
                else:
                    prev_pipelinestage = self.pipelinestages[current_index_pipelinestage_index - 1]
                    prev_pipelinestage_phase = prev_pipelinestage.phases[index]

                    if prev_pipelinestage_phase.active == True:
                        prev_pipelinestage_phase_closed = False
                    else:
                        prev_pipelinestage_phase_closed = True

                if prev_pipelinestage_phase_closed == False:
                    break

                if pipelinestage.resourcetype == 'CPU':
                    current_resources = cpu_resources
                else:
                    current_resources = gpu_resources

                none_executing = True

                for resource in current_resources:
                    workitem = resource.get_workitem(pipelinestage.resourcetype)

                    #if workitem != None:
                    #   print ('close phases', pipelinestage.name, resource.id, workitem.version, index, pipelinestage_index)

                    if workitem != None and int(workitem.version) == current_index_pipelinestage_index:
                        none_executing = False
                        break

                if none_executing == True:
                    phase.close_phase()

                current_index_pipelinestage_index += 1

            index += 1

    def add_executor (self, workitem, resource, now):
        pipelinestage = self.pipelinestages[int (workitem.version)]
        pipelinestage.add_executor (workitem, resource, now)
        if int (workitem.version) > 0:
            prev_pipelinestage = self.pipelinestages[int (workitem.version) - 1]
            prev_pipelinestage.remove_output (workitem)

    def remove_executor (self, workitem, resource, now):
        pipelinestage = self.pipelinestages[int (workitem.version)]
        pipelinestage.add_output(workitem)
        pipelinestage.remove_executor (workitem, resource, now)

    def add_workitem_queue_old (self, workitem, current_time):
        pipelinestage_remove = None
        pipelinestage_add = None

        if workitem == None:
            print ('invalid workitem')
            return
        #print ('add_workitem_queue', workitem.id, workitem.iscomplete)

        if workitem.iscomplete == True:
            if int (workitem.version) >= len (self.pipelinestages) - 1:# should check if stage type matches
                pipelinestage_remove = self.pipelinestages[int (workitem.version)]
            else:
                pipelinestage_remove = self.pipelinestages[int (workitem.version)]
                pipelinestage_add = self.pipelinestages[int (workitem.version) + 1]

            if pipelinestage_remove != None:
                remove_phase, remove_phase_index = pipelinestage_remove.get_phase (workitem)
                #print ('add_workitem_queue1', remove_phase, remove_phase_index)
                if remove_phase != None:
                    remove_phase.remove_workitem (current_time, workitem)

            if pipelinestage_add != None:
                if pipelinestage_add.resourcetype != pipelinestage_remove.resourcetype: #this shouldn't be done here
                    add_phase = pipelinestage_add.get_phase_index (remove_phase_index)
                    add_phase.add_workitem (workitem, current_time)
        else:
            pipelinestage_add = self.pipelinestages[int(workitem.version)]
            if pipelinestage_add.index == 0:
                if len (pipelinestage_add.phases) == 0:
                    self.build_phases(2)
                else:
                    if pipelinestage_add.phases[-2].active == False and pipelinestage_add.phases[-2].complete == True:
                        self.build_phases(1)
            else:
                print ('add_workitem_queue', 'oops', pipelinestage_add.index)
            pipelinestage_add.add_new_workitem (workitem, current_time)

    def add_workitem_queue (self, workitem, current_time, new_workitem_index = -1):
        pipelinestage_remove = None
        pipelinestage_add = None

        if workitem == None:
            print ('invalid workitem')
            return
        #print ('add_workitem_queue', workitem.id, workitem.iscomplete)

        if workitem.iscomplete == True:
            if int (workitem.version) >= len (self.pipelinestages) - 1:# should check if stage type matches
                pipelinestage_remove = self.pipelinestages[int (workitem.version)]
            else:
                pipelinestage_remove = self.pipelinestages[int (workitem.version)]
                pipelinestage_add = self.pipelinestages[int (workitem.version) + 1]

            if pipelinestage_remove != None:
                remove_phase, remove_phase_index = pipelinestage_remove.get_phase (workitem)
                #print ('add_workitem_queue1', remove_phase.pipelinestage, remove_phase_index)
                if remove_phase != None:
                    remove_phase.remove_workitem (current_time, workitem)

            if pipelinestage_add != None:
                if pipelinestage_add.resourcetype != pipelinestage_remove.resourcetype: #this shouldn't be done here
                    add_phase = pipelinestage_add.get_phase_index (remove_phase_index)
                    add_phase.add_workitem (workitem, current_time)
        else:
            pipelinestage_add = self.pipelinestages[int(workitem.version)]
            if pipelinestage_add.index == 0:
                if new_workitem_index == -1:
                    pipelinestage_add.add_new_workitem(workitem, current_time, self.last_first_phase_closed_index)
                else:
                    pipelinestage_add.add_new_workitem(workitem, current_time, new_workitem_index - 1)
            else:
                print ('add_workitem_queue', 'oops', pipelinestage_add.index)


    def check_new_workitem_index (self):
        first_pipelinestage = self.pipelinestages[0]

        #print ('check_new_workitem_index ()', self.last_first_phase_closed_index, first_pipelinestage.phases[self.last_first_phase_closed_index + 1].total_count)
        if first_pipelinestage.phases[self.last_first_phase_closed_index + 1].total_count == self.batchsize:
            return self.last_first_phase_closed_index + 2

        return self.last_first_phase_closed_index + 1

    def build_phases (self):
        for i in range(0, self.no_of_columns):
            index = 0
            prev_resourcetype = None
            while True:
                if index >= len (self.pipelinestages):
                    print ('build phases', i, index, len (self.pipelinestages))
                    break
                current_stage = self.pipelinestages[index]
                if prev_resourcetype == None or current_stage.resourcetype != prev_resourcetype:
                    current_stage.create_phase ()
                    prev_resourcetype = current_stage.resourcetype
                index += 1

    def print_stage_queue_data (self):
        for index in range (0, len(self.pipelinestages[0].phases)):
            for pipelinestage in self.pipelinestages:
                pipelinestage.print_data (index)
            print ("####################")

    def print_stage_prediction_data (self):
        for index in range (0, len(self.pipelinestages[0].phases)):
            for pipelinestage in self.pipelinestages:
                pipelinestage.print_data (index)

    '''
    def encode_pipeline_stages (self, pipelinestages):
        output = ""
        index = 0
        for pipelinestage in pipelinestages:
            if index == len (pipelinestages) - 1:
                output += str (pipelinestage.get_name())
            else:
                output += str (pipelinestage.get_name())
                output += ":"
            index += 1
        return output
    '''

    def get_all_pipelinestages (self):
        return self.pipelinestages

    def get_pipelinestage (self, current, resourcetype):
        if current == None:
            if self.pipelinestages[0].get_resourcetype () != resourcetype:
                return None

        elif current.get_resourcetype () == resourcetype or \
            current.get_index () == len (self.pipelinestages) - 1:
            return None

        ret = None

        if current == None:
            index = 0
        else:
            index = current.get_index () + 1

        ret = self.pipelinestages[index]

        '''
        while index < len (self.pipelinestages) and \
            self.pipelinestages[index].get_resourcetype() == resourcetype:
            ret.append (self.pipelinestages[index])
            index += 1
        '''

        return ret

if __name__ == "__main__":
    pipelinefile = sys.argv[1]
    p = PipelineManager(pipelinefile)
    p.parse_pipelines ()
    p.print_data ()
