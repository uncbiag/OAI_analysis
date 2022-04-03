import yaml
import sys
from parslfluxsim.resources_sim import Resource
import statistics
import copy
import math
from plots.plot_prediction_sim import plot_prediction_sim
from plots.plot_prediction_sim import plot_prediction_sim_0
from plots.plot_prediction_sim import plot_prediction
from plots.plot_prediction_sim import plot_prediction_idle_periods

class Phase:
    def __init__(self, pipelinestage, index, resourcetype, rmanager, target, pipelinestage_index):
        self.pipelinestage = pipelinestage
        self.pipelinestage_index = pipelinestage_index
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
        self.queue_snapshots = {}
        if self.pipelinestage_index == 0:
            self.queue_snapshots[str(0.0)] = self.target
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
        self.p_outputs = []
        self.persistent_p_outputs = []
        self.predictions = {}
        self.pthrottle_idle_period = -1
        self.pthrottle_idle_start_time = -1

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
            print(self.pipelinestage, 'activate phase', self.index)
        self.current_count += 1
        self.pcurrent_count = self.current_count
        self.total_count += 1
        self.ptotal_count = self.total_count
        self.add_timestamps[workitem.id + ':' + str (currenttime)] = self.current_count
        self.workitems.append(workitem.id)
        if self.pipelinestage_index > 0:
            self.queue_snapshots[str(currenttime)] = self.current_count
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

        if self.pipelinestage_index == 0:
            self.queue_snapshots[str(currenttime)] = self.target - self.total_complete
        else:
            self.queue_snapshots[str(currenttime)] = self.current_count
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
              self.first_resource_release_time, self.endtime, self.total_count, self.index)

    def get_queued_work (self, rmanager, resourcetype, current_time):
        queued_work = 0
        # work_done, [start_time, fractional_finish_time, whole_finish_time]
        #print ('get_queued_work ()', self.pqueued)
        for key in self.pqueued:
            queued_work += (1 - self.pqueued[key][0])
        self.pqueued_work = queued_work

        if self.pqueued_work <= 0.0:
            queued_work = 0
            for executor in self.current_executors:
                executor = rmanager.get_resource(executor)
                work_left = executor.get_work_left(resourcetype, current_time)
                if work_left == None:
                    print('workitem doesnt exist', executor)
                    continue

                queued_work += work_left

            self.queued_work = queued_work

            return self.queued_work
        else:
            return self.pqueued_work

    def print_prediction_data (self):
        for prediction_key in self.predictions.keys ():
            prediction = self.predictions[prediction_key]
            output = ""
            for item in prediction:
                output += " " + str(item)

            print('prediction ()', prediction_key, output)

    def print_data (self):
        print('print_data ()', self.pipelinestage, self.starttime, self.first_workitem_completion_time,
              self.first_resource_release_time, self.endtime, self.total_count,
              float(self.endtime) - float(self.starttime),
              float(self.endtime) - float(self.first_resource_release_time),
              len(self.predictions)
              )
        self.print_prediction_data()
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

        new_phase = Phase(self.name.split(':')[0], len (self.phases), self.resourcetype, self.rmanager, self.batchsize, self.index)
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
        #print(self.name, 'add executor', now, workitem.id, index, self.phases[index].total_complete)

    def remove_executor (self, workitem, resource, now):
        current_phase, index = self.get_phase(workitem)
        if current_phase == None:
            print('remove_executor', workitem.id, self.name, 'not found')
            return
        current_phase.remove_executor (resource, now)
        #print(self.name, 'remove executor', workitem.id, index)

    def add_output (self, workitem):
        phase, index = self.get_phase(workitem)
        phase.outputs.append(workitem.endtime)
        phase.persistent_outputs.append (workitem.endtime)
        phase.p_outputs.append(workitem.endtime)
        phase.persistent_p_outputs.append(workitem.endtime)

    def remove_output (self, workitem):
        phase, index = self.get_phase(workitem)
        #print (phase.pipelinestage, phase.index, phase.outputs)
        #print (phase.pipelinestage, phase.index, phase.p_outputs)
        phase.outputs.pop (0)
        phase.p_outputs.pop (0)

    def get_current_throughput (self, rmanager, phase_index):
        current_executors = self.phases[phase_index].current_executors

        thoughput_list = []
        for resource_id in current_executors:
            resource = rmanager.get_resource (resource_id)
            if self.resourcetype == 'CPU':
                resource_name = resource.cpu.name
            else:
                resource_name = resource.gpu.name

            exectime = resource.get_exectime(self.name, self.resourcetype) #TODO: get latest info, not long executimes history
            if exectime == 0:
                exectime = rmanager.get_exectime(resource_name, self.name)
                if exectime == 0:
                    print('get_current_throughput ()', 'exectime does not exist')
                    continue
            thoughput_list.append (1 / exectime)

        return sum (thoughput_list)

    def get_free_resource_throughput(self, rmanager, free_resources):
        thoughput_list = []
        for resource_id in free_resources:
            resource = rmanager.get_resource (resource_id)
            if self.resourcetype == 'CPU':
                resource_name = resource.cpu.name
            else:
                resource_name = resource.gpu.name
            exectime = resource.get_exectime(self.name, self.resourcetype)  # TODO: get latest info, not long executimes history
            if exectime == 0:
                exectime = rmanager.get_exectime(resource_name, self.name)
                if exectime == 0:
                    print('get_current_throughput ()', 'exectime does not exist')
                    continue
            thoughput_list.append(1 / exectime)

        return sum(thoughput_list)

    def get_resource_service_rate (self, rmanager):
        pipelinestage_resources = rmanager.get_resources_type(self.resourcetype)

        #print ('get_resource_service_rate ()', self.name)
        all_resource_service_rates = []
        for resource in pipelinestage_resources:
            exectime = resource.get_exectime(self.name, self.resourcetype)
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
            exectime = resource.get_exectime(self.name, self.resourcetype)
            if exectime == 0:
                print('exectime does not exist')
                continue
            all_resource_service_rates.append(1 / exectime)

        if len (all_resource_service_rates) <= 0:
            return 0

        self.avg_resource_service_rate = sum(all_resource_service_rates) / len (all_resource_service_rates)

        #print ('get_avg_resource_service_rate ()', self.avg_resource_service_rate)

        return self.avg_resource_service_rate

    def get_time_required_1 (self, rmanager, target, current_time, phase_index):
        #print('get_time_required_1 ()', self.name, target, current_time)

        pipelinestage_resources = rmanager.get_resources_type(self.resourcetype)
        phase = self.phases[phase_index]
        resource_data = {}
        end_times = {}
        total_queued_work = 0

        #print('get_time_required_1 ()', phase.pcurrent_executors)
        #print('get_time_required_1 ()', phase.pqueued)

        for resource_id in phase.pcurrent_executors:
            resource = rmanager.get_resource(resource_id)
            queued_work = resource.get_work_left(self.resourcetype, current_time)
            #print(resource_id, queued_work)
            if (queued_work == None or queued_work == 0) and resource_id in phase.pqueued.keys():
                queued_work = 1 - phase.pqueued[resource_id][0]

            exectime = resource.get_exectime(self.name, self.resourcetype)
            service_rate = 1 / exectime
            times_required = queued_work / service_rate
            #queued_work, finish_time, service_rate, resource.id
            resource_data[resource.id] = [queued_work, current_time + times_required, service_rate, resource.id]
            total_queued_work += queued_work

        #print('get_time_required_1 ()', total_queued_work)

        resource_index = 0
        while resource_index < len(pipelinestage_resources):
            resource = pipelinestage_resources[resource_index]
            if resource.id in phase.pcurrent_executors and resource_data[resource.id][0] > 0.0:
                resource_index += 1
                continue

            if total_queued_work + 1 > target:
                resource_data[resource.id] = [0, -1, 0, resource.id]
                end_times[resource.id] = current_time
            else:
                queued_work = 1
                exectime = resource.get_exectime(self.name, self.resourcetype)
                service_rate = 1 / exectime
                times_required = queued_work / service_rate
                # queued_work, finish_time, service_rate, resource.id
                resource_data[resource.id] = [queued_work, current_time + times_required, service_rate, resource.id]
                total_queued_work += queued_work

            resource_index += 1

        #print('get_time_required_1 ()', resource_data)
        #print('get_time_required_1 ()', end_times)

        while True:
            sorted_resource_items = sorted(resource_data.items(), key=lambda kv: kv[1][1])
            selected_resource_key = None
            for resource in sorted_resource_items:
                end_time = resource[1][1]
                if end_time > 0.0:
                    selected_resource_key = resource[0]
                    break
            if end_time <= 0.0:
                break
            #print(target, total_queued_work, end_time, sorted_resource_items)
            #print(end_times)

            resource = resource_data[selected_resource_key]

            resource[0] = 0.0
            end_times[selected_resource_key] = resource[1]

            if total_queued_work + 1 <= target:
                resource[0] = 1
                total_queued_work += 1
                resource[1] += (resource[0] / resource[2])
            else:
                resource[0] = 0
                resource[1] = -1

        sorted_end_times = sorted(end_times.items(), key=lambda kv: kv[1])

        final_end_times = {}

        for end_time in sorted_end_times:
            final_end_times[end_time[0]] = end_time[1]

        final_end_time = list(final_end_times.values())[-1]
        resource_release_time = list(final_end_times.values())[0]

        #print('get_time_required_1 ()', final_end_times)

        return final_end_time, resource_release_time, final_end_times

    def get_time_required_2 (self, rmanager, target, current_time, phase_index):
        #print ('get_time_required_2 ()', self.name, target, current_time)

        pipelinestage_resources = rmanager.get_resources_type(self.resourcetype)
        phase = self.phases[phase_index]
        resource_data = {}
        end_times = {}
        total_queued_work = 0

        #print ('get_time_required_2 ()', phase.pcurrent_executors)
        #print ('get_time_required_2 ()', phase.pqueued)

        for resource_id in phase.pcurrent_executors:
            resource = rmanager.get_resource (resource_id)
            queued_work = resource.get_work_left(self.resourcetype, current_time)
            #print (resource_id, queued_work)
            if (queued_work == None or queued_work == 0) and resource_id in phase.pqueued.keys ():
                queued_work = 1 - phase.pqueued[resource_id][0]

            exectime = resource.get_exectime(self.name, self.resourcetype)
            service_rate = 1 / exectime
            times_required = queued_work / service_rate
            #queued_work, finish_time, service_rate, resource.id
            resource_data[resource.id] = [queued_work, current_time + times_required, service_rate, resource.id]
            total_queued_work += queued_work

        #print ('get_time_required_2 ()', total_queued_work)

        resource_index = 0
        while resource_index < len (pipelinestage_resources):
            resource = pipelinestage_resources[resource_index]
            if resource.id in phase.pcurrent_executors and resource_data[resource.id][0] > 0.0:
                resource_index += 1
                continue
            elif resource.id in phase.pcurrent_executors:
                phase.p_outputs.append(current_time)
                phase.persistent_p_outputs.append(current_time)

            if total_queued_work + 1 > target:
                resource_data[resource.id] = [0, -1, 0, resource.id]
                end_times[resource.id] = current_time
            else:
                queued_work = 1
                exectime = resource.get_exectime(self.name, self.resourcetype)
                service_rate = 1 / exectime
                times_required = queued_work / service_rate
                # queued_work, finish_time, service_rate, resource.id
                resource_data[resource.id] = [queued_work, current_time + times_required, service_rate, resource.id]
                total_queued_work += queued_work

            resource_index += 1

        #print ('get_time_required_2 ()', resource_data)
        #print ('get_time_required_2 ()', end_times)

        while True:
            sorted_resource_items = sorted(resource_data.items(), key=lambda kv: kv[1][1])
            selected_resource_key = None
            for resource in sorted_resource_items:
                end_time = resource[1][1]
                if end_time > 0.0:
                    selected_resource_key = resource[0]
                    break
            if end_time <= 0.0:
                break
            #print (target, total_queued_work, end_time, sorted_resource_items)
            #print (end_times)

            resource = resource_data[selected_resource_key]

            resource[0] = 0.0
            end_times[selected_resource_key] = resource[1]

            phase.p_outputs.append (resource[1])
            phase.persistent_p_outputs.append (resource[1])

            if total_queued_work + 1 <= target:
                resource[0] = 1
                total_queued_work += 1
                resource[1] += (resource[0] / resource[2])
            else:
                resource[0] = 0
                resource[1] = -1

        sorted_end_times = sorted(end_times.items(), key=lambda kv: kv[1])

        final_end_times = {}

        for end_time in sorted_end_times:
            final_end_times[end_time[0]] = end_time[1]

        final_end_time = list (final_end_times.values ())[-1]
        resource_release_time = list (final_end_times.values ())[0]

        #print ('get_time_required_2 ()', final_end_times)

        return final_end_time, resource_release_time, final_end_times

    def get_time_required_3 (self, rmanager, target, current_time, phase_index, previous_phase_outputs, starttimes):
        #print ('get_time_required_3 ()', self.name, target, current_time)

        pipelinestage_resources = rmanager.get_resources_type(self.resourcetype)
        phase = self.phases[phase_index]
        resource_data = {}
        end_times = {}

        #print ('get_time_required_3 ()', phase.pcurrent_executors)
        #print ('get_time_required_3 ()', phase.pqueued)
        #print('get_time_required_3 ()', previous_phase_outputs)

        for resource_id in phase.pcurrent_executors:
            resource = rmanager.get_resource (resource_id)
            queued_work = resource.get_work_left(self.resourcetype, current_time)
            #print (resource_id, queued_work)
            if (queued_work == None or queued_work == 0) and resource_id in phase.pqueued.keys ():
                queued_work = 1 - phase.pqueued[resource_id][0]

            exectime = resource.get_exectime(self.name, self.resourcetype)
            service_rate = 1 / exectime
            times_required = queued_work / service_rate
            #queued_work, finish_time, service_rate, resource.id
            resource_data[resource.id] = [queued_work, current_time + times_required, service_rate, resource.id]

        resource_index = 0
        while resource_index < len (pipelinestage_resources):
            resource = pipelinestage_resources[resource_index]
            if resource.id in phase.pcurrent_executors and resource_data[resource.id][0] > 0.0:
                resource_index += 1
                continue
            elif resource.id in phase.pcurrent_executors:
                phase.p_outputs.append(current_time)
                phase.persistent_p_outputs.append(current_time)

            if len (previous_phase_outputs) <= 0:
                resource_data[resource.id] = [0, -1, 0, resource.id]
                if starttimes == None:
                    end_times[resource.id] = current_time
                else:
                    end_times[resource.id] = starttimes[resource.id]
                continue

            exectime = resource.get_exectime(self.name, self.resourcetype)
            service_rate = 1 / exectime
            times_required = 1 / service_rate

            if starttimes == None:
                prev_end_time = current_time
            else:
                prev_end_time = starttimes[resource.id]
            if prev_end_time > previous_phase_outputs[0]:
                resource_data[resource.id] = [1, prev_end_time + times_required, service_rate, resource.id]
            else:
                resource_data[resource.id] = [1, previous_phase_outputs[0] + times_required, service_rate, resource.id]

            resource_index += 1
            previous_phase_outputs.pop(0)

        #print ('get_time_required_3 ()', resource_data)
        #print ('get_time_required_3 ()', end_times)
        #print ('get_time_required_3 ()', previous_phase_outputs)

        while True:
            sorted_resource_items = sorted(resource_data.items(), key=lambda kv: kv[1][1])
            selected_resource_key = None
            for resource in sorted_resource_items:
                end_time = resource[1][1]
                if end_time > 0.0:
                    selected_resource_key = resource[0]
                    break
            if end_time <= 0.0:
                break
            #print (target, end_time, sorted_resource_items)
            #print (previous_phase_outputs)
            #print (end_times)

            resource = resource_data[selected_resource_key]
            resource[0] = 0.0
            end_times[selected_resource_key] = resource[1]

            phase.p_outputs.append (resource[1])
            phase.persistent_p_outputs.append (resource[1])

            if len (previous_phase_outputs) > 0:
                resource[0] = 1
                if resource[1] > previous_phase_outputs[0]:
                    resource[1] += (resource[0] / resource[2])
                else:
                    resource[1] = previous_phase_outputs[0] + (resource[0] / resource[2])
                previous_phase_outputs.pop(0)
            else:
                resource[0] = 0
                resource[1] = -1

        sorted_end_times = sorted(end_times.items(), key=lambda kv: kv[1])

        final_end_times = {}

        for end_time in sorted_end_times:
            final_end_times[end_time[0]] = end_time[1]

        final_end_time = list(final_end_times.values())[-1]
        resource_release_time = list(final_end_times.values())[0]

        #print('get_time_required_3 ()', final_end_times)

        return final_end_time, resource_release_time, final_end_times

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
                resource_service_rate = 1 / resource.get_exectime(self.name, self.resourcetype)
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

        #print ('------------------------------------')
        #print ('remaining idle times ()', idle_times)
        #print ('work done ()', work_done)
        #print ('output times ()', output_times)
        #print ('fraction work ()', fractional_work)
        #print('------------------------------------')

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
                resource_service_rate = 1 / resource.get_exectime(self.name, self.resourcetype)
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

        #print('------------------------------------')
        #print('remaining idle times ()', idle_times)
        #print('work done ()', work_done)
        #print('output times ()', output_times)
        #print('fraction work ()', fractional_work)
        #print('------------------------------------')

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
        self.prediction_times = []
        self.prediction_idle_periods = {}
        self.max_images = max_images

    def get_pct_complete_no_prediction (self):
        return self.pipelinestages[-1].phases[0].total_complete / self.max_images * 100

    def get_weighted_performance_to_cost_ratio_ranking (self, rmanager, resourcetype, resource_ids):

        no_of_pipelinestages = 0
        for pipelinestage in self.pipelinestages:
            if pipelinestage.resourcetype == resourcetype:
                no_of_pipelinestages += 1

        weighted_performance_to_cost_ratio = {}
        weights = rmanager.get_pipelinestage_weights(resourcetype, no_of_pipelinestages)
        pipelinestage_weights = {}
        pipelinestageindex = 0
        for pipelinestage in self.pipelinestages:
            pipelinestage_weights[pipelinestage.name] = weights[pipelinestageindex]

        print (resource_ids)

        for resource_id in resource_ids:
            resource = rmanager.get_resource (resource_id)
            if resourcetype == 'CPU':
                resource_name = resource.cpu.name
            else:
                resource_name = resource.gpu.name

            performance_to_cost_ratio = 0
            for pipelinestage in self.pipelinestages:
                if pipelinestage.resourcetype == resourcetype:
                    exectime = resource.get_exectime(pipelinestage.name, resourcetype)
                    if exectime == 0:
                        exectime = rmanager.get_exectime (resource_name, pipelinestage.name)
                        throughput = 1 / exectime
                    else:
                        throughput = 1 / exectime
                    performance_to_cost_ratio += throughput / resource.get_cost(resourcetype) * pipelinestage_weights[pipelinestage.name]
            if performance_to_cost_ratio == 0:
                weighted_performance_to_cost_ratio[resource.id] = 0
            else:
                weighted_performance_to_cost_ratio[resource.id] = performance_to_cost_ratio

        weighted_performance_to_cost_ratio_ranking = dict(sorted(weighted_performance_to_cost_ratio.items(), key=lambda item: item[1]))

        return weighted_performance_to_cost_ratio_ranking

    def get_weighted_performance_to_cost_ratio_ranking_all (self, rmanager, resourcetype):
        no_of_pipelinestages = 0
        for pipelinestage in self.pipelinestages:
            if pipelinestage.resourcetype == resourcetype:
                no_of_pipelinestages += 1

        weighted_performance_to_cost_ratio = {}
        weights = rmanager.get_pipelinestage_weights(resourcetype, no_of_pipelinestages)
        pipelinestage_weights = {}
        pipelinestageindex = 0
        for pipelinestage in self.pipelinestages:
            pipelinestage_weights[pipelinestage.name] = weights[pipelinestageindex]

        resourcetnames = rmanager.get_resource_names (resourcetype)

        for resource_name in resourcetnames:
            performance_to_cost_ratio = 0
            for pipelinestage in self.pipelinestages:
                if pipelinestage.resourcetype == resourcetype:
                    exectime = rmanager.get_exectime (resource_name, pipelinestage.name)
                    if exectime == 0:
                        exectime = rmanager.get_exectime(resource_name, pipelinestage.name)
                        throughput = 1 / exectime
                    else:
                        throughput = 1 / exectime
                    performance_to_cost_ratio += throughput / rmanager.get_resourcetype_info (resource_name, 'cost')

            if performance_to_cost_ratio == 0:
                weighted_performance_to_cost_ratio[resource_name] = 0
            else:
                weighted_performance_to_cost_ratio[resource_name] = performance_to_cost_ratio

        weighted_performance_to_cost_ratio_ranking = dict(sorted(weighted_performance_to_cost_ratio.items(), key=lambda item: item[1], reverse= True))

        return weighted_performance_to_cost_ratio_ranking

    def scale_down_configuration (self, rmanager, pipelinestageindex, overallocation, throughput, available_resources):
        to_be_deleted = []
        target_throughput = float((1 - overallocation)) * throughput
        pipelinestage = self.pipelinestages[pipelinestageindex]

        while True:
            weighted_pcr_ranking = self.get_weighted_performance_to_cost_ratio_ranking (rmanager, pipelinestage.resourcetype, available_resources)

            print ('scale_down_configuration', weighted_pcr_ranking)

            removed_at_least_one = False
            for resource_id in weighted_pcr_ranking.keys ():
                resource = rmanager.get_resource (resource_id)
                if pipelinestage.resourcetype == 'CPU':
                    resource_name = resource.cpu.name
                else:
                    resource_name = resource.gpu.name
                exectime = resource.get_exectime(pipelinestage.name, pipelinestage.resourcetype)
                if exectime == 0:
                    exectime = rmanager.get_exectime(resource_name, pipelinestage.name)
                    resource_throughput = 1 / exectime
                else:
                    resource_throughput = 1 / exectime

                throughput = throughput - resource_throughput

                print ('scale_down_configuration', resource_id, resource_name, resource_throughput, throughput, target_throughput)

                if throughput - target_throughput >= 0:
                    to_be_deleted.append(resource_id)
                    available_resources.remove (resource_id)
                    removed_at_least_one = True
                    break
                else:
                    throughput += resource_throughput

            if removed_at_least_one == False:
                break
        return to_be_deleted

    def scale_up_configuration(self, rmanager, pipelinestageindex, input_pressure, throughput):
        to_be_added = {}
        target_throughput = input_pressure - throughput
        pipelinestage = self.pipelinestages[pipelinestageindex]

        added_throughput = 0

        while True:
            weighted_pcr_ranking = self.get_weighted_performance_to_cost_ratio_ranking_all(rmanager, pipelinestage.resourcetype)

            print ('scale_up_configuration', weighted_pcr_ranking)

            for resource_name in weighted_pcr_ranking.keys ():
                exectime = rmanager.get_exectime(resource_name, pipelinestage.name)

                if exectime == 0:
                    exectime = rmanager.get_exectime(resource_name, pipelinestage.name)
                    resource_throughput = 1 / exectime
                else:
                    resource_throughput = 1 / exectime

                resource_available = rmanager.request_resource (resource_name)

                print('scale_up_configuration', resource_name, resource_throughput, resource_available, throughput, added_throughput, target_throughput)

                if resource_available == True:
                    if added_throughput + resource_throughput > added_throughput:
                        if resource_name not in to_be_added:
                            to_be_added[resource_name] = 1
                            added_throughput = added_throughput + resource_throughput
                        else:
                            to_be_added[resource_name] += 1
                            added_throughput = added_throughput + resource_throughput
                        break

            if added_throughput >= target_throughput:
                break
        return to_be_added

    def scale_up_configuration_limit (self, rmanager, pipelinestageindex, input_pressure, throughput, limit):
        to_be_added = {}
        target_throughput = input_pressure - throughput
        pipelinestage = self.pipelinestages[pipelinestageindex]
        added_throughput = 0
        total_acquired = 0

        while True:
            weighted_pcr_ranking = self.get_weighted_performance_to_cost_ratio_ranking_all(rmanager,
                                                                                           pipelinestage.resourcetype)

            print('scale_up_configuration', weighted_pcr_ranking)


            for resource_name in weighted_pcr_ranking.keys():
                exectime = rmanager.get_exectime(resource_name, pipelinestage.name)

                if exectime == 0:
                    exectime = rmanager.get_exectime(resource_name, pipelinestage.name)
                    resource_throughput = 1 / exectime
                else:
                    resource_throughput = 1 / exectime

                resource_available = rmanager.request_resource(resource_name)

                print('scale_up_configuration', resource_name, resource_throughput, resource_available, throughput,
                      target_throughput)

                if resource_available == True:
                    if added_throughput + resource_throughput > added_throughput:
                        if resource_name not in to_be_added:
                            to_be_added[resource_name] = 1
                            added_throughput = added_throughput + resource_throughput
                        else:
                            to_be_added[resource_name] += 1
                            added_throughput = added_throughput + resource_throughput
                        total_acquired += 1
                        break

            if total_acquired >= limit:
                break

            if added_throughput >= target_throughput:
                break
        return to_be_added

    def calculate_pipeline_stats (self, rmanager, current_time, free_cpus, free_gpus):
        throughputs = {}
        pending_workloads = {}
        computation_pressures= {}
        available_resources = {}
        max_throughputs = {}

        total_cpu_input_pressure = 0
        total_gpu_input_pressure = 0
        total_cpu_throughput = 0
        total_gpu_throughput = 0


        current_free_cpus = copy.deepcopy(free_cpus)
        current_free_gpus = copy.deepcopy(free_gpus)

        pipelinestageindex = len(self.pipelinestages) - 1

        while pipelinestageindex >= 0:
            pipelinestage = self.pipelinestages[pipelinestageindex]
            if pipelinestage.resourcetype == 'CPU':
                free_resources = current_free_cpus
            else:
                free_resources = current_free_gpus

            throughputs[str(pipelinestageindex)] = pipelinestage.get_current_throughput(rmanager, 0)

            if pipelinestageindex != 0:
                if pipelinestage.resourcetype == 'CPU':
                    pending_workloads[str(pipelinestageindex)] = pipelinestage.phases[0].current_count - len(
                        pipelinestage.phases[0].current_executors) + pipelinestage.phases[0].get_queued_work(rmanager,
                                                                                                             'CPU',
                                                                                                             current_time)
                else:
                    pending_workloads[str(pipelinestageindex)] = pipelinestage.phases[0].current_count - len(
                        pipelinestage.phases[0].current_executors) + pipelinestage.phases[0].get_queued_work(rmanager,
                                                                                                             'GPU',
                                                                                                             current_time)

            if pipelinestage.resourcetype == 'CPU':
                if pipelinestageindex < len(self.pipelinestages) - 1:
                    total_gpu_input_pressure += throughputs[str(pipelinestageindex)]
            else:
                if pipelinestageindex < len(self.pipelinestages) - 1:
                    total_cpu_input_pressure += throughputs[str(pipelinestageindex)]

            if pipelinestage.resourcetype == 'CPU':
                total_cpu_throughput += throughputs[str(pipelinestageindex)]
            else:
                total_gpu_throughput += throughputs[str(pipelinestageindex)]

            if pipelinestageindex == 0:
                computation_pressures[str(pipelinestageindex)] = [0, throughputs[str(pipelinestageindex)]]
            else:
                prev_pipelinestage = self.pipelinestages[pipelinestageindex - 1]
                throughputs[str(pipelinestageindex - 1)] = prev_pipelinestage.get_current_throughput(rmanager, 0)
                computation_pressures[str(pipelinestageindex)] = [throughputs[str(pipelinestageindex - 1)],
                                                                  throughputs[str(pipelinestageindex)]]

            if throughputs[str(pipelinestageindex)] <= 0:
                if computation_pressures[str(pipelinestageindex)][0] <= 0:
                    available_resources[str(pipelinestageindex)] = []
                    max_throughputs[str(pipelinestageindex)] = 0
                else:
                    available_resources[str(pipelinestageindex)] = copy.deepcopy(free_resources)
                    max_throughputs[str(pipelinestageindex)] = pipelinestage.get_free_resource_throughput(rmanager,
                                                                                                          free_resources)
                    free_resources.clear()
            else:
                available_resources[str(pipelinestageindex)] = copy.deepcopy(
                    pipelinestage.phases[0].current_executors) + copy.deepcopy(free_resources)
                max_throughputs[str(pipelinestageindex)] = pipelinestage.get_free_resource_throughput(rmanager,
                                                                                                      free_resources) + pipelinestage.get_current_throughput(
                    rmanager, 0)
                free_resources.clear()

            pipelinestageindex -= 1

        return throughputs, pending_workloads, computation_pressures, available_resources, max_throughputs, total_cpu_input_pressure, total_gpu_input_pressure, total_cpu_throughput, total_gpu_throughput

    def reconfiguration_down(self, rmanager, current_time, free_cpus, free_gpus):
        print('empty cpus', len(free_cpus))
        print('empty gpus', len(free_gpus))

        throughputs, pending_workloads, computation_pressures, available_resources, max_throughputs, total_cpu_input_pressure, total_gpu_input_pressure, total_cpu_throughput, total_gpu_throughput = self.calculate_pipeline_stats (rmanager, current_time, free_cpus, free_gpus)


        overallocations = {}
        underallocations = {}

        overallocations[str(0)] = 0.0
        underallocations[str(0)] = 0.0

        gpus_to_be_added = {}
        cpus_to_be_added = {}
        gpus_to_be_dropped = []
        cpus_to_be_dropped = []

        pipelinestageindex = 1
        while pipelinestageindex < len(self.pipelinestages):
            pipelinestage = self.pipelinestages[pipelinestageindex]

            if pipelinestage.resourcetype == 'CPU':
                total_throughput = total_cpu_throughput
            else:
                total_throughput = total_gpu_throughput

            if max_throughputs[str(pipelinestageindex)] == 0:
                overallocations[str(pipelinestageindex)] = 0.0
                if throughputs[str(pipelinestageindex - 1)] > 0:
                    underallocations[str(pipelinestageindex)] = 1.0
                else:
                    underallocations[str(pipelinestageindex)] = 0.0
            elif throughputs[str(str(pipelinestageindex))] < max_throughputs[str(pipelinestageindex)]:
                underallocations[str(pipelinestageindex)] = 0.0
                if computation_pressures[str(pipelinestageindex)][0] > 0:
                    overallocations[str(pipelinestageindex)] = (max_throughputs[str(pipelinestageindex)] -
                                                                computation_pressures[str(pipelinestageindex)][0]) / \
                                                               max_throughputs[str(pipelinestageindex)]
                else:
                    overallocations[str(pipelinestageindex)] = (max_throughputs[str(pipelinestageindex)] - throughputs[
                        str(pipelinestageindex)]) / max_throughputs[str(pipelinestageindex)]
            else:
                overallocations[str(pipelinestageindex)] = 0.0
                if computation_pressures[str(pipelinestageindex)][0] > 0:
                    underallocations[str(pipelinestageindex)] = (computation_pressures[str(pipelinestageindex)][0] -
                                                                 max_throughputs[str(pipelinestageindex)]) / \
                                                                computation_pressures[str(pipelinestageindex)][0]
                else:
                    underallocations[str(pipelinestageindex)] = 0.0

            if str(pipelinestageindex) in underallocations and underallocations[str(pipelinestageindex)] >= 1.0 and total_throughput <= 0:
                print('reconfiguration up 1 ()', pipelinestage.name, underallocations, overallocations,
                      computation_pressures[str(pipelinestageindex)], max_throughputs[str(pipelinestageindex)],
                      available_resources, pending_workloads)
                to_be_added = self.scale_up_configuration(rmanager, pipelinestageindex,
                                                          computation_pressures[str(pipelinestageindex)][0],
                                                          max_throughputs[str(pipelinestageindex)])
                if pipelinestage.resourcetype == 'CPU':
                    print('CPUs to be added', to_be_added)
                    cpus_to_be_added[str(pipelinestageindex)] = to_be_added
                else:
                    print('GPUs to be added', to_be_added)
                    gpus_to_be_added[str(pipelinestageindex)] = to_be_added

            '''
            elif str(pipelinestageindex) in underallocations and underallocations[str(pipelinestageindex)] > 0.0 and total_throughput > 0 and total_throughput == throughputs[str(pipelinestageindex)]:
                print('reconfiguration up 2 ()', pipelinestage.name, underallocations, overallocations,
                      computation_pressures[str(pipelinestageindex)], max_throughputs[str(pipelinestageindex)],
                      available_resources, pending_workloads)
                to_be_added = self.scale_up_configuration(rmanager, pipelinestageindex,
                                                          computation_pressures[str(pipelinestageindex)][0],
                                                          max_throughputs[str(pipelinestageindex)])
                if pipelinestage.resourcetype == 'CPU':
                    print('CPUs to be added', to_be_added)
                    cpus_to_be_added[str(pipelinestageindex)] = to_be_added
                else:
                    print('GPUs to be added', to_be_added)
                    gpus_to_be_added[str(pipelinestageindex)] = to_be_added
            '''

            if str(pipelinestageindex) in overallocations and overallocations[str(pipelinestageindex)] > 0:
                print('reconfiguration ()', pipelinestage.name, overallocations[str(pipelinestageindex)],
                      computation_pressures[str(pipelinestageindex)], max_throughputs[str(pipelinestageindex)],
                      available_resources)
                to_be_dropped = self.scale_down_configuration(rmanager, pipelinestageindex,
                                                              overallocations[str(pipelinestageindex)],
                                                              max_throughputs[str(pipelinestageindex)],
                                                              available_resources[str(pipelinestageindex)])
                if pipelinestage.resourcetype == 'CPU':
                    print('CPUs to be dropped', to_be_dropped)
                    cpus_to_be_dropped.extend(to_be_dropped)
                else:
                    print('GPUs to be dropped', to_be_dropped)
                    gpus_to_be_dropped.extend(to_be_dropped)

            pipelinestageindex += 1

        if total_gpu_throughput <= 0 and total_gpu_input_pressure <= 0:
            gpu_weighted_pcr_ranking = self.get_weighted_performance_to_cost_ratio_ranking(rmanager, 'GPU', free_gpus)

            to_be_dropped = []

            for gpu_id in gpu_weighted_pcr_ranking.keys():
                gpu = rmanager.get_resource(gpu_id)
                if rmanager.get_availability(gpu.gpu.name) > 0.8:
                    to_be_dropped.append(gpu.id)
            print('GPUs to be dropped', to_be_dropped)
            gpus_to_be_dropped.extend(to_be_dropped)

        if total_cpu_throughput <= 0 and total_cpu_input_pressure <= 0:
            cpu_weighted_pcr_ranking = self.get_weighted_performance_to_cost_ratio_ranking(rmanager, 'CPU', free_cpus)

            to_be_dropped = []

            for cpu_id in cpu_weighted_pcr_ranking.keys():
                cpu = rmanager.get_resource(cpu_id)
                if rmanager.get_availability(cpu.cpu.name) > 0.8:
                    to_be_dropped.append(cpu.id)
            print('CPUs to be dropped', to_be_dropped)
            cpus_to_be_dropped.extend (to_be_dropped)

        print('total throughput', total_cpu_throughput, total_gpu_throughput)
        print('total input pressure', total_cpu_input_pressure, total_gpu_input_pressure)
        print('computation pressures', computation_pressures)
        print('max throughputs', max_throughputs)
        print ('available resources', available_resources)

        return cpus_to_be_dropped, gpus_to_be_dropped, cpus_to_be_added, gpus_to_be_added


    def reconfiguration_up_down_underallocations (self, rmanager, current_time, free_cpus, free_gpus):
        print('reconfiguration_up_down_underallocations ()', free_cpus, free_gpus)

        throughputs, pending_workloads, computation_pressures, available_resources, max_throughputs, total_cpu_input_pressure, total_gpu_input_pressure, total_cpu_throughput, total_gpu_throughput = self.calculate_pipeline_stats(rmanager, current_time, free_cpus, free_gpus)

        underallocations = {}
        underallocations[str(0)] = 0.0

        gpus_to_be_added = {}
        cpus_to_be_added = {}

        pipelinestageindex = 1
        while pipelinestageindex < len(self.pipelinestages):
            pipelinestage = self.pipelinestages[pipelinestageindex]

            if pipelinestage.resourcetype == 'CPU':
                total_throughput = total_cpu_throughput
            else:
                total_throughput = total_gpu_throughput

            if max_throughputs[str(pipelinestageindex)] == 0:
                if throughputs[str(pipelinestageindex - 1)] > 0:
                    underallocations[str(pipelinestageindex)] = 1.0
                else:
                    underallocations[str(pipelinestageindex)] = 0.0
            elif throughputs[str(str(pipelinestageindex))] < max_throughputs[str(pipelinestageindex)]:
                underallocations[str(pipelinestageindex)] = 0.0
            else:
                if computation_pressures[str(pipelinestageindex)][0] > 0:
                    underallocations[str(pipelinestageindex)] = (computation_pressures[str(pipelinestageindex)][0] -
                                                                 max_throughputs[str(pipelinestageindex)]) / \
                                                                 computation_pressures[str(pipelinestageindex)][0]
                else:
                    underallocations[str(pipelinestageindex)] = 0.0

            pending_workitems = pipelinestage.phases[0].current_count - len(available_resources[str(pipelinestageindex)])

            if str(pipelinestageindex) in underallocations and pending_workitems > 0 and ((underallocations[str(pipelinestageindex)] >= 1.0 and total_throughput <= 0) or (underallocations[str(pipelinestageindex)] > 0 and throughputs[str(pipelinestageindex)] == total_throughput)):
                print('reconfiguration_up_down_underallocations 1 ()', pipelinestage.name, underallocations,
                      computation_pressures[str(pipelinestageindex)], max_throughputs[str(pipelinestageindex)],
                      available_resources, pending_workloads)
                to_be_added = self.scale_up_configuration_limit(rmanager, pipelinestageindex,
                                                                computation_pressures[str(pipelinestageindex)][0],
                                                                max_throughputs[str(pipelinestageindex)], pending_workitems)
                if pipelinestage.resourcetype == 'CPU':
                    print('CPUs to be added', to_be_added)
                    cpus_to_be_added[str(pipelinestageindex)] = to_be_added
                else:
                    print('GPUs to be added', to_be_added)
                    gpus_to_be_added[str(pipelinestageindex)] = to_be_added
            elif str(pipelinestageindex) in underallocations and pending_workitems > 0 and (underallocations[str(pipelinestageindex)] <= 0.0 and total_throughput <= 0):
                print('reconfiguration_up_down_underallocations 2 ()', pipelinestage.name, underallocations,
                      computation_pressures[str(pipelinestageindex)], max_throughputs[str(pipelinestageindex)],
                      available_resources, pending_workloads)
                to_be_added = self.scale_up_configuration_limit(rmanager, pipelinestageindex,
                                                                computation_pressures[str(pipelinestageindex)][0],
                                                                max_throughputs[str(pipelinestageindex)],
                                                                pending_workitems)
                if pipelinestage.resourcetype == 'CPU':
                    print('CPUs to be added', to_be_added)
                    cpus_to_be_added[str(pipelinestageindex)] = to_be_added
                else:
                    print('GPUs to be added', to_be_added)
                    gpus_to_be_added[str(pipelinestageindex)] = to_be_added

            pipelinestageindex += 1

        return cpus_to_be_added, gpus_to_be_added

    def reconfiguration_up_down_overallocations (self, rmanager, current_time, free_cpus, free_gpus):

        print('reconfiguration_up_down_overallocations ()', free_cpus, free_gpus)

        throughputs, pending_workloads, computation_pressures, available_resources, max_throughputs, total_cpu_input_pressure, total_gpu_input_pressure, total_cpu_throughput, total_gpu_throughput = self.calculate_pipeline_stats(rmanager, current_time, free_cpus, free_gpus)

        overallocations = {}

        overallocations[str(0)] = 0.0

        gpus_to_be_dropped = []
        cpus_to_be_dropped = []

        pipelinestageindex = 1
        while pipelinestageindex < len(self.pipelinestages):
            pipelinestage = self.pipelinestages[pipelinestageindex]

            if pipelinestage.resourcetype == 'CPU':
                total_throughput = total_cpu_throughput
            else:
                total_throughput = total_gpu_throughput

            if max_throughputs[str(pipelinestageindex)] == 0:
                overallocations[str(pipelinestageindex)] = 0.0
            elif throughputs[str(str(pipelinestageindex))] < max_throughputs[str(pipelinestageindex)]:
                if computation_pressures[str(pipelinestageindex)][0] > 0:
                    overallocations[str(pipelinestageindex)] = (max_throughputs[str(pipelinestageindex)] -
                                                                computation_pressures[str(pipelinestageindex)][0]) / \
                                                               max_throughputs[str(pipelinestageindex)]
                else:
                    overallocations[str(pipelinestageindex)] = (max_throughputs[str(pipelinestageindex)] - throughputs[
                        str(pipelinestageindex)]) / max_throughputs[str(pipelinestageindex)]
            else:
                overallocations[str(pipelinestageindex)] = 0.0

            if str(pipelinestageindex) in overallocations and overallocations[str(pipelinestageindex)] > 0:
                print('reconfiguration ()', pipelinestage.name, overallocations[str(pipelinestageindex)],
                      computation_pressures[str(pipelinestageindex)], max_throughputs[str(pipelinestageindex)],
                      available_resources)
                to_be_dropped = self.scale_down_configuration(rmanager, pipelinestageindex,
                                                              overallocations[str(pipelinestageindex)],
                                                              max_throughputs[str(pipelinestageindex)],
                                                              available_resources[str(pipelinestageindex)])
                if pipelinestage.resourcetype == 'CPU':
                    print('CPUs to be dropped', to_be_dropped)
                    cpus_to_be_dropped.extend(to_be_dropped)
                else:
                    print('GPUs to be dropped', to_be_dropped)
                    gpus_to_be_dropped.extend(to_be_dropped)

            pipelinestageindex += 1

        return cpus_to_be_dropped, gpus_to_be_dropped

    def reconfiguration_drop (self, rmanager, current_time, free_cpus, free_gpus):

        gpus_to_be_dropped = []
        cpus_to_be_dropped = []

        print ('reconfiguration_drop ()', free_cpus, free_gpus)

        throughputs, pending_workloads, computation_pressures, available_resources, max_throughputs, total_cpu_input_pressure, total_gpu_input_pressure, total_cpu_throughput, total_gpu_throughput = self.calculate_pipeline_stats(rmanager, current_time, free_cpus, free_gpus)

        if total_gpu_throughput <= 0 and total_gpu_input_pressure <= 0:
            gpu_weighted_pcr_ranking = self.get_weighted_performance_to_cost_ratio_ranking(rmanager, 'GPU', free_gpus)

            to_be_dropped = []

            for gpu_id in gpu_weighted_pcr_ranking.keys():
                gpu = rmanager.get_resource(gpu_id)
                if rmanager.get_availability(gpu.gpu.name) > 0.8:
                    to_be_dropped.append(gpu.id)
            print('GPUs to be dropped', to_be_dropped)
            gpus_to_be_dropped.extend(to_be_dropped)

        if total_cpu_throughput <= 0 and total_cpu_input_pressure <= 0:
            cpu_weighted_pcr_ranking = self.get_weighted_performance_to_cost_ratio_ranking(rmanager, 'CPU', free_cpus)

            to_be_dropped = []

            for cpu_id in cpu_weighted_pcr_ranking.keys():
                cpu = rmanager.get_resource(cpu_id)
                if rmanager.get_availability(cpu.cpu.name) > 0.8:
                    to_be_dropped.append(cpu.id)
            print('CPUs to be dropped', to_be_dropped)
            cpus_to_be_dropped.extend(to_be_dropped)

        print('total throughput', total_cpu_throughput, total_gpu_throughput)
        print('total input pressure', total_cpu_input_pressure, total_gpu_input_pressure)
        print('computation pressures', computation_pressures)
        print('max throughputs', max_throughputs)
        print('available resources', available_resources)

        return cpus_to_be_dropped, gpus_to_be_dropped



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
        for key in end_times.keys ():
            resource_id = key
            resource_finish_time = end_times[key]

            idle_periods[resource_id] = [[resource_finish_time, finish_time, finish_time - resource_finish_time]]
            #idle_periods[resource_id] = finish_time - resource_finish_time

        idle_periods = dict(sorted(idle_periods.items(), key=lambda item: item[1][-1][2], reverse=True))

        return idle_periods

    def prediction_reset (self, start_index):
        for pipelinestage in self.pipelinestages:
            for current_index in range(start_index, len (pipelinestage.phases)):
                pipelinestage.phases[current_index].prediction_reset ()


    def check_throttle_idleness(self, last_phase_closed_index, current_time, pending_workitem):
        pipelinestage_index = int(pending_workitem.version) + 1
        phase_index = pending_workitem.phase_index

        pipelinestage = self.pipelinestages[pipelinestage_index]
        phase = pipelinestage.phases[phase_index]

        if phase.pthrottle_idle_period != -1:
            if phase.pthrottle_idle_start_time == -1:
                phase.pthrottle_idle_start_time = current_time

            if phase.pthrottle_idle_start_time + phase.pthrottle_idle_period > current_time:
                #print ('check_throttle_idleness', pipelinestage.name, current_time, phase.pthrottle_idle_start_time, phase.pthrottle_idle_period)
                return False, phase.pthrottle_idle_start_time + phase.pthrottle_idle_period
            else:
                return True, phase.pthrottle_idle_start_time + phase.pthrottle_idle_period

        return True, -1

    def check_workitem_waiting_idleness (self, rmanager, resource_id, resourcetype, last_phase_closed_index, target_pipelinestage, current_time):
        current_phase_index = last_phase_closed_index + 1
        print ('into check_workitem_waiting_idleness ()')
        pipelinestage_index = 0
        for pipelinestage in self.pipelinestages:
            if pipelinestage.resourcetype == resourcetype:
                if pipelinestage_index + 2 < len (self.pipelinestages):
                    next_sametype_pipelinestage = self.pipelinestages[pipelinestage_index + 2]
                    print('check_workitem_waiting_idleness ()', pipelinestage.name, current_phase_index, next_sametype_pipelinestage.phases[current_phase_index].total_count)
                    if next_sametype_pipelinestage.phases[current_phase_index].total_count == 0:
                        next_pipelinestage = self.pipelinestages[pipelinestage_index + 1]
                        print('check_workitem_waiting_idleness ()', next_sametype_pipelinestage.name,
                              current_phase_index)
                        print ('check_workitem_waiting_idleness ()', next_pipelinestage.phases[current_phase_index].current_executors)
                        min_time_left = -1
                        for executor_id in next_pipelinestage.phases[current_phase_index].current_executors:
                            executor = rmanager.get_resource (executor_id)
                            time_left = executor.get_time_left (next_pipelinestage.resourcetype, current_time)
                            if time_left == None:
                                continue
                            if min_time_left == -1:
                                min_time_left = time_left
                            elif min_time_left > time_left:
                                min_time_left = time_left

                        print('check_workitem_waiting_idleness ()',min_time_left)

                        target_resource = rmanager.get_resource (resource_id)
                        target_exectime = target_resource.get_exectime(target_pipelinestage)
                        if min_time_left != -1 and target_exectime > min_time_left:
                            print ('check_workitem_waiting_idleness ()', pipelinestage.name, current_phase_index, min_time_left, resource_id, target_exectime)
                            return False
                        break
                    else:
                        return False
            pipelinestage_index += 1

        return True

    def predict_execution_fixed (self, rmanager, current_time, batchsize, last_phase_index_closed, no_of_prediction_phases):
        index = last_phase_index_closed + 1

        total_cpu_idle_periods = {}
        total_gpu_idle_periods = {}

        print(index, '###############################################################')

        self.prediction_reset (index)

        prediction_key = str (current_time)

        self.prediction_times.append(prediction_key)

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
                        #print ('setting pstarttime')
                    if phase.pfirst_workitem_completion_time == -1:
                        phase.pfirst_workitem_completion_time = phase.pstarttime + 1 / avg_resource_service_rate  # use fastest
                        #print ('setting pfirst_workitem_completion_time')

                    finish_time, first_resource_release_time, end_times = pipelinestage.get_time_required_2(rmanager, work_to_be_done, current_time, index)

                    phase.pendtime = finish_time

                    phase.end_times = end_times

                    if phase.pfirst_resource_release_time == -1:
                        phase.pfirst_resource_release_target = batchsize - (len (pipelinestage_resources) - 1)
                        phase.pfirst_resource_release_time = first_resource_release_time
                        #print ('setting pfirst_resource_release_time')

                    phase.predictions[prediction_key] = [current_time, index, pipelinestage.name.split(':')[0],
                                                         phase.starttime, phase.pstarttime, phase.pendtime,
                                                         phase.pending_output, phase.pfirst_workitem_completion_time, phase.pfirst_resource_release_time]
                else:
                    prev_sametype_phase = prev_prev_pipelinestage_phase

                    if phase.pstarttime == -1:
                        #print ('setting pstarttime', current_time, prev_pipelinestage_phase.pfirst_workitem_completion_time)
                        if prev_sametype_phase == None:
                            if current_time > prev_pipelinestage_phase.pfirst_workitem_completion_time:
                                phase.pstarttime = current_time
                            else:
                                phase.pstarttime = prev_pipelinestage_phase.pfirst_workitem_completion_time
                        else:
                            #print ('setting pstarttime', prev_sametype_phase.pfirst_resource_release_time)
                            if prev_pipelinestage_phase.pfirst_workitem_completion_time > prev_sametype_phase.pfirst_resource_release_time:
                                phase.pstarttime = prev_pipelinestage_phase.pfirst_workitem_completion_time
                            else:
                                phase.pstarttime = prev_sametype_phase.pfirst_resource_release_time

                            if current_time > phase.pstarttime:
                                phase.pstarttime = current_time

                    if phase.pfirst_workitem_completion_time == -1:
                        phase.pfirst_workitem_completion_time = phase.pstarttime + 1 / avg_resource_service_rate
                        # print ('setting pfirst_workitem_completion_time')

                    starttimes = None
                    if prev_sametype_phase != None:
                        starttimes = prev_sametype_phase.end_times
                        if prev_pipelinestage_phase.pfirst_workitem_completion_time > prev_sametype_phase.pfirst_resource_release_time:
                            idle_periods = self.get_idle_periods(prev_sametype_phase.end_times, phase.pstarttime)
                            idle_period_start = list(idle_periods.values())[0][-1][0]

                            print('idle period 1', prev_sametype_phase.pipelinestage, [idle_period_start, phase.pstarttime],
                                  idle_periods)

                            if phase.resourcetype == 'CPU':
                                cpu_idle_periods.append([idle_period_start, phase.pstarttime, idle_periods])
                            else:
                                gpu_idle_periods.append([idle_period_start, phase.pstarttime, idle_periods])

                    if prev_pipelinestage.all_resource_service_rate > pipelinestage.all_resource_service_rate:
                        finish_time, first_resource_release_time, end_times = pipelinestage.get_time_required_3(rmanager,
                                                                                                     work_to_be_done,
                                                                                                     current_time,
                                                                                                     index, prev_pipelinestage_phase.p_outputs, starttimes)

                        phase.pendtime = finish_time
                        phase.end_times = end_times

                        if phase.pfirst_resource_release_time == -1:
                            #print ('setting pfirst_resource_release_time')
                            phase.pfirst_resource_release_target = batchsize - (len(pipelinestage_resources) - 1)
                            phase.pfirst_resource_release_time = first_resource_release_time

                        if pipelinestage_index == len(self.pipelinestages) - 1:
                            idle_periods = self.get_idle_periods (phase.end_times, phase.pendtime)
                            idle_period_start = list(idle_periods.values())[0][-1][0]
                            idle_period_end = phase.pendtime

                            print('idle period 2', phase.pipelinestage,
                                  [idle_period_start, idle_period_end],
                                  idle_periods)

                            if phase.resourcetype == 'CPU':
                                cpu_idle_periods.append([idle_period_start, idle_period_end, idle_periods])
                            else:
                                gpu_idle_periods.append([idle_period_start, idle_period_end, idle_periods])

                        phase.predictions[prediction_key] = [current_time, index, pipelinestage.name.split(':')[0],
                                                             phase.starttime, phase.pstarttime, phase.pendtime,
                                                             phase.pending_output,
                                                             phase.pfirst_workitem_completion_time,
                                                             phase.pfirst_resource_release_time]
                    else:
                        if phase.pstarttime < current_time:
                            queued_finish_time, _, _ = pipelinestage.get_time_required_1(rmanager,
                                                                                         queued_work + phase.pcurrent_count - len (phase.pcurrent_executors),
                                                                                         current_time, index)
                            fastest_finish_time, _, _ = pipelinestage.get_time_required_1(rmanager, work_to_be_done,
                                                                                          current_time, index)
                        else:
                            queued_finish_time, _, _ = pipelinestage.get_time_required_1(rmanager,
                                                                                         queued_work + phase.pcurrent_count - len(
                                                                                             phase.pcurrent_executors),
                                                                                         phase.pstarttime, index)
                            fastest_finish_time, _, _ = pipelinestage.get_time_required_1(rmanager, work_to_be_done,
                                                                                          phase.pstarttime, index)
                        #print(phase.pipelinestage, queued_finish_time)

                        #print(phase.pipelinestage, queued_finish_time, finish_time)

                        if queued_finish_time >= prev_pipelinestage_phase.pendtime:
                            finish_time, first_resource_release_time, end_times = pipelinestage.get_time_required_3(rmanager,
                                                                                                     work_to_be_done,
                                                                                                     current_time,
                                                                                                     index, prev_pipelinestage_phase.p_outputs, starttimes)
                            phase.pendtime = finish_time
                            phase.end_times = end_times

                            if phase.pfirst_resource_release_time == -1:
                                phase.pfirst_resource_release_target = batchsize - (len(pipelinestage_resources) - 1)
                                phase.pfirst_resource_release_time = first_resource_release_time
                        else:
                            estimated_finish_time = prev_pipelinestage_phase.pendtime + 1 / avg_resource_service_rate

                            idle_period = estimated_finish_time - fastest_finish_time
                            if phase.pstarttime < current_time:
                                idle_period_start = current_time
                            else:
                                idle_period_start = phase.pstarttime

                            starttimes = {}
                            idle_periods = {}
                            for resource in pipelinestage_resources:
                                if resource.id in phase.pcurrent_executors:
                                    starttimes[resource.id] = idle_period_start + idle_period #TODO: use a function get_queued_work_resource() that looks into either get_work_left or pqueued {}
                                    if resource.id in phase.pqueued.keys ():
                                        idle_periods[resource.id] = [[phase.pqueued[resource.id][1][2], idle_period_start + idle_period, idle_period_start + idle_period - phase.pqueued[resource.id][1][2]]]
                                    else:
                                        time_left = resource.get_time_left(phase.resourcetype, current_time)
                                        idle_periods[resource.id] = [[current_time + time_left, idle_period_start + idle_period, idle_period_start + idle_period - (current_time + time_left)]]
                                else:
                                    starttimes[resource.id] = idle_period_start + idle_period
                                    idle_periods[resource.id] = [[idle_period_start, idle_period_start + idle_period, idle_period]]

                            print('idle period 3', phase.pipelinestage, [idle_period_start,
                                                                         idle_period_start + idle_period],
                                                                         idle_periods)

                            if phase.pstarttime >= current_time:
                                phase.pstarttime = idle_period_start + idle_period
                                phase.pfirst_workitem_completion_time = phase.pstarttime + (1/avg_resource_service_rate)

                            if index == last_phase_index_closed + 1:
                                phase.pthrottle_idle_period = idle_period

                            if phase.resourcetype == 'CPU':
                                cpu_idle_periods.append(
                                    [idle_period_start, idle_period_start + idle_period, idle_periods])
                            else:
                                gpu_idle_periods.append(
                                    [idle_period_start, idle_period_start + idle_period, idle_periods])

                            finish_time, first_resource_release_time, end_times = pipelinestage.get_time_required_3(rmanager,
                                                                                                     work_to_be_done,
                                                                                                     current_time,
                                                                                                     index, prev_pipelinestage_phase.p_outputs, starttimes)
                            phase.pendtime = finish_time
                            phase.end_times = end_times

                            if phase.pfirst_resource_release_time == -1:
                                phase.pfirst_resource_release_time = first_resource_release_time


                        phase.predictions[prediction_key] = [current_time, index, pipelinestage.name.split(':')[0],
                                                             phase.starttime, phase.pstarttime, phase.pendtime,
                                                             phase.pending_output,
                                                             phase.pfirst_workitem_completion_time,
                                                             phase.pfirst_resource_release_time]

                    if pipelinestage_index == len (self.pipelinestages) - 1:
                        idle_periods = self.get_idle_periods(prev_pipelinestage_phase.end_times, phase.pendtime)

                        print ('idle period 4', prev_pipelinestage_phase.pipelinestage, [prev_pipelinestage_phase.pendtime, phase.pendtime], idle_periods)

                        if prev_pipelinestage_phase.resourcetype == 'CPU':
                            cpu_idle_periods.append ([prev_pipelinestage_phase.pendtime, phase.pendtime, idle_periods])
                        else:
                            gpu_idle_periods.append ([prev_pipelinestage_phase.pendtime, phase.pendtime, idle_periods])


                prediction = phase.predictions[prediction_key]

                output = ""
                for item in prediction:
                    output += " " + str(item)

                print ('predict_execution ()', output)

                prev_prev_pipelinestage = prev_pipelinestage
                prev_pipelinestage = pipelinestage
                prev_prev_pipelinestage_phase = prev_pipelinestage_phase
                prev_pipelinestage_phase = phase
                pipelinestage_index += 1

            print(index, '###############################################################')

            print('cpu idle periods before precompute ()', index, cpu_idle_periods)
            print('gpu idle periods before precompute ()', index, gpu_idle_periods)

            self.pre_compute_prediction(rmanager, batchsize, cpu_idle_periods, gpu_idle_periods, index + 1, prediction_key)

            print('cpu idle periods post precompute ()', index, cpu_idle_periods)
            print('gpu idle periods post precompute ()', index, gpu_idle_periods)

            if str(index) not in self.prediction_idle_periods.keys():
                self.prediction_idle_periods[str(index)] = [cpu_idle_periods, gpu_idle_periods]

            print(index, '***************************************************************')

            index += 1
            current_time = phase.pendtime


        self.prediction_reset(last_phase_index_closed + 1)

    def pre_compute_prediction (self, rmanager, batchsize, cpu_idle_periods, gpu_idle_periods, phase_index, prediction_key):

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

                #print('pre_compute_prediction ()', pipelinestage_name, fractional_work_dict[pipelinestage_name])
                #print('pre_compute_prediction ()', pipelinestage_name, current_phase.persistent_p_outputs)

                next_phase = None
                if pipelinestage_index < len (self.pipelinestages) - 1:
                    next_phase = self.pipelinestages[pipelinestage_index + 1].phases[current_phase_index]

                if current_phase.pstarttime == -1:
                    current_phase.pstarttime = starttimes_dict[pipelinestage_name]

                if work_done_dict[pipelinestage_name] > 0.0:
                    if current_phase.pfirst_workitem_completion_time == -1:
                        current_phase.pfirst_workitem_completion_time = current_phase.persistent_p_outputs[0]

                    current_phase.ptotal_complete += work_done_dict[pipelinestage_name]

                    current_phase.pending_output = batchsize - current_phase.ptotal_complete

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

                if current_phase.pendtime != -1:
                    current_phase.predictions[prediction_key] = [prediction_key, current_phase_index, current_phase.pipelinestage.split(':')[0],
                                                                 current_phase.starttime, current_phase.pstarttime, current_phase.pendtime,
                                                                 current_phase.pending_output,
                                                                 current_phase.pfirst_workitem_completion_time,
                                                                 current_phase.pfirst_resource_release_time]

                if work_done_dict[pipelinestage_name] <= 0.0:
                    break

                pipelinestage_index += 1

            if pipelinestage_index == 0:
                break

            current_phase_index += 1

    def calculate_early_computation (self, rmanager, batchsize, cpu_idle_periods, gpu_idle_periods, phase_index, first_index):
        from operator import itemgetter
        sorted_cpu_idle_periods = sorted(cpu_idle_periods, key=itemgetter(0))
        sorted_gpu_idle_periods = sorted(gpu_idle_periods, key=itemgetter(0))

        #print('sorted cpu idle periods ()', phase_index, sorted_cpu_idle_periods)
        #print('sorted gpu idle periods ()', phase_index, sorted_gpu_idle_periods)

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
                #print ('calculate_early_computation', 'target is zero', pipelinestage.name)
                continue

            idle_period_index = 0

            while idle_period_index < len (idle_periods) and work_done_dict[pipelinestage.name] < target:
                idle_period = idle_periods[idle_period_index]

                #print ('calculate_early_computation', pipelinestage.name, idle_period_index, target, idle_period)

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

        #print('new sorted cpu idle periods ()', sorted_cpu_idle_periods)
        #print('new sorted gpu idle periods ()', sorted_gpu_idle_periods)

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

    def print_stage_queue_data_3 (self, actual_idle_periods):
        plot_prediction_idle_periods (actual_idle_periods, self.prediction_idle_periods)

    def print_stage_queue_data_2 (self, rmanager):
        plot_data = {}
        for pipelinestage in self.pipelinestages:
            plot_data[pipelinestage.name] = []
            for phase in pipelinestage.phases:
                plot_data[pipelinestage.name].append ([phase.queue_snapshots, phase.starttime, phase.endtime, phase.predictions])

        plot_prediction_sim_0 (rmanager, plot_data, self.prediction_times, self.batchsize)


    def print_stage_queue_data_1 (self):
        plot_data = {}
        for pipelinestage in self.pipelinestages:
            plot_data[pipelinestage.name] = []
            for phase in pipelinestage.phases:
                plot_data[pipelinestage.name].append ([phase.queue_snapshots, phase.starttime, phase.endtime, phase.predictions])

        plot_prediction_sim (plot_data, self.prediction_times,self.batchsize)


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
