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
        #print(self.pipelinestage, 'add workitem', self.index, currenttime, workitem.id, self.current_count, self.total_count, self.total_complete)

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

        pipelinestage_resources = self.rmanager.get_resources_type(self.resourcetype, active=True)
        if self.total_complete == self.target - len (pipelinestage_resources) + 1:
            self.first_resource_release_time = workitem.endtime
            self.pfirst_resource_release_time = workitem.endtime

        if self.pipelinestage_index == 0:
            self.queue_snapshots[str(currenttime)] = self.target - self.total_complete
        else:
            self.queue_snapshots[str(currenttime)] = self.current_count
        #print(self.pipelinestage, 'remove workitem', self.index, currenttime, workitem.id, self.current_count, self.total_count, self.total_complete)

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
                executor = rmanager.get_resource(executor, active=True)
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
    def __init__ (self, stageindex, name, resourcetype, rmanager, target):
        index = 0
        self.name = name
        self.index = stageindex
        self.resourcetype = resourcetype
        self.phases = []
        self.rmanager = rmanager
        self.batchsize = target
        self.exec_parents = []
        self.exec_children = []
        self.data_parents = []
        self.data_children = []

    def add_dependency_child (self, child, type):
        if type == 'exec':
            self.exec_children.append (child)
        else:
            self.data_children.append(child)

    def add_dependency_parent (self, parent, type):
        if type == 'exec':
            self.exec_parents.append (parent)
        else:
            self.data_parents.append(parent)

    def get_children (self, type):
        if type == 'exec':
            return self.exec_children
        else:
            return self.data_children

    def get_parent (self, type):
        if type == 'exec':
            return self.exec_parents
        else:
            return self.data_parents

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

    def get_current_throughput (self, phase_index):
        current_executors = self.phases[phase_index].current_executors

        thoughput_list = []
        for resource_id in current_executors:
            resource = self.rmanager.get_resource (resource_id, active=True)
            if self.resourcetype == 'CPU':
                resource_name = resource.cpu.name
            else:
                resource_name = resource.gpu.name

            exectime = resource.get_exectime(self.name, self.resourcetype) #TODO: get latest info, not long executimes history
            if exectime == 0:
                exectime = self.rmanager.get_exectime(resource_name, self.name)
                if exectime == 0:
                    #print('get_current_throughput ()', 'exectime does not exist')
                    continue
            thoughput_list.append (1 / exectime)

        return sum (thoughput_list)

    def get_free_resource_throughput(self, rmanager, free_resources):
        thoughput_list = []
        for resource_id in free_resources:
            resource = rmanager.get_resource (resource_id, active=True)
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


    def print_data (self):
        print (self.name, self.index, self.resourcetype, self.data_parents, self.exec_parents, self.data_children, self.exec_children)

class PipelineManager:
    def __init__ (self, pipelinefile, budget, batchsize, max_images):
        self.pipelinefile = pipelinefile
        self.pipelinestages = []
        self.pipelinestages_dict = {}
        self.batchsize = batchsize
        self.budget = budget
        self.last_last_phase_closed_index = -1
        self.last_first_phase_closed_index = -1
        self.no_of_columns = int(math.ceil(max_images / batchsize))
        self.prediction_times = []
        self.prediction_idle_periods = {}
        self.max_images = max_images
        self.throughput_record = {}

    def record_throughput (self, env):
        for pipelinestage in self.pipelinestages:
            current_throughput = pipelinestage.get_current_throughput (0)
            if str(pipelinestage.index) not in self.throughput_record.keys ():
                self.throughput_record[str(pipelinestage.index)] = []
                self.throughput_record[str (pipelinestage.index)].append ([env.now, current_throughput])
            else:
                self.throughput_record[str(pipelinestage.index)].append([env.now, current_throughput])
    def get_throughput_record (self):
        return self.throughput_record

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
            resource = rmanager.get_resource (resource_id, active=True)
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
                    ret = rmanager.get_resourcetype_info (resource_name, 'cost', 'on_demand')
                    if ret != None:
                        on_demand_cost = ret
                    else:
                        on_demand_cost = 0

                    ret = rmanager.get_resourcetype_info(resource_name, 'cost', 'spot')
                    if ret != None:
                        spot_cost = ret
                    else:
                        spot_cost = 0

                    total_cost = on_demand_cost + spot_cost

                    performance_to_cost_ratio += throughput / total_cost

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
                resource = rmanager.get_resource (resource_id, active=True)
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

    def scale_down_configuration_imbalance_limit (self, rmanager, pipelinestageindex, overallocation, throughput, available_resources, imbalance_limit):
        to_be_deleted = []
        target_throughput = float((1 - overallocation)) * throughput
        target_throughput_ub = target_throughput + (target_throughput * imbalance_limit/100)
        target_throughput_lb = target_throughput - (target_throughput * imbalance_limit / 100)
        pipelinestage = self.pipelinestages[pipelinestageindex]

        while True:
            if throughput >= target_throughput_lb and throughput <= target_throughput_ub:
                print('scale_down_configuration: within range', throughput, target_throughput_lb, target_throughput_ub)
                break
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

                print ('scale_down_configuration', resource_id, resource_name, resource_throughput, throughput, target_throughput_lb, target_throughput_ub)

                if throughput >= target_throughput_lb:
                    to_be_deleted.append(resource_id)
                    available_resources.remove(resource_id)
                    removed_at_least_one = True
                    break
                else:
                    throughput += resource_throughput

                '''
                if throughput - target_throughput >= 0:
                    to_be_deleted.append(resource_id)
                    available_resources.remove (resource_id)
                    removed_at_least_one = True
                    break
                else:
                    throughput += resource_throughput
                '''

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

    def scale_up_configuration_limit_imbalance_limit (self, rmanager, pipelinestageindex, input_pressure, throughput, resource_limit, imbalance_limit):
        to_be_added = {}
        target_throughput = input_pressure - throughput
        target_throughput = input_pressure
        target_throughput_ub = target_throughput + (target_throughput * imbalance_limit / 100)
        target_throughput_lb = target_throughput - (target_throughput * imbalance_limit / 100)
        pipelinestage = self.pipelinestages[pipelinestageindex]
        added_throughput = 0
        total_acquired = 0

        while True:
            if throughput >= target_throughput_lb and throughput <=target_throughput_ub:
                print('scale_up_configuration_limit: within range', throughput, target_throughput_lb, target_throughput_ub)
                break
            weighted_pcr_ranking = self.get_weighted_performance_to_cost_ratio_ranking_all(rmanager,
                                                                                           pipelinestage.resourcetype)

            print('scale_up_configuration', weighted_pcr_ranking)


            added_at_least_once = False
            for resource_name in weighted_pcr_ranking.keys():
                exectime = rmanager.get_exectime(resource_name, pipelinestage.name)

                if exectime == 0:
                    exectime = rmanager.get_exectime(resource_name, pipelinestage.name)
                    resource_throughput = 1 / exectime
                else:
                    resource_throughput = 1 / exectime

                resource_available = rmanager.request_resource(resource_name)

                print('scale_up_configuration', resource_name, resource_throughput, resource_available, throughput, target_throughput_lb, target_throughput_ub)

                if resource_available == True:
                    if resource_throughput + throughput > throughput:
                        if resource_throughput + throughput <= target_throughput_ub:
                            if resource_name not in to_be_added:
                                to_be_added[resource_name] = 1
                                throughput = throughput + resource_throughput
                            else:
                                to_be_added[resource_name] += 1
                                throughput = throughput + resource_throughput
                            total_acquired += 1
                            added_at_least_once = True
                            break


            if total_acquired >= resource_limit:
                break

            if added_at_least_once == False:
                break
        return to_be_added

    def scale_up_configuration_limit (self, rmanager, pipelinestageindex, input_pressure, throughput, resource_limit, throughput_limit):
        to_be_added = {}
        target_throughput = input_pressure - throughput
        if throughput_limit != 0:
            target_throughput = throughput_limit - throughput
        pipelinestage = self.pipelinestages[pipelinestageindex]
        added_throughput = 0
        total_acquired = 0

        if target_throughput <= 0:
            return to_be_added

        while True:
            weighted_pcr_ranking = self.get_weighted_performance_to_cost_ratio_ranking_all(rmanager,
                                                                                           pipelinestage.resourcetype)

            print('scale_up_configuration_limit', weighted_pcr_ranking)

            for resource_name in weighted_pcr_ranking.keys():
                exectime = rmanager.get_exectime(resource_name, pipelinestage.name)

                if exectime == 0:
                    exectime = rmanager.get_exectime(resource_name, pipelinestage.name)
                    resource_throughput = 1 / exectime
                else:
                    resource_throughput = 1 / exectime

                resource_available = rmanager.request_resource(resource_name)

                print('scale_up_configuration_limit', resource_name, resource_throughput, resource_available, throughput, target_throughput, resource_limit)

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

            if total_acquired >= resource_limit:
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
        upcoming_throughputs = {}

        total_cpu_input_pressure = 0
        total_gpu_input_pressure = 0
        total_cpu_throughput = 0
        total_gpu_throughput = 0


        current_free_cpus = copy.deepcopy(free_cpus)
        current_free_gpus = copy.deepcopy(free_gpus)

        pipelinestageindex = len(self.pipelinestages) - 1

        while pipelinestageindex >= 0:
            pipelinestage = self.pipelinestages[pipelinestageindex]
            if pipelinestageindex - 2 >= 0:
                prev_sametype_pipelinestage = self.pipelinestages[pipelinestageindex - 2]
            else:
                prev_sametype_pipelinestage = None
            if pipelinestage.resourcetype == 'CPU':
                free_resources = current_free_cpus
            else:
                free_resources = current_free_gpus

            throughputs[str(pipelinestageindex)] = pipelinestage.get_current_throughput(0)

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
                throughputs[str(pipelinestageindex - 1)] = prev_pipelinestage.get_current_throughput(0)
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
                                                                                                      free_resources) + pipelinestage.get_current_throughput(0)
                free_resources.clear()

            if prev_sametype_pipelinestage != None:
                upcoming_throughputs[str(pipelinestageindex)] = pipelinestage.get_free_resource_throughput(rmanager, prev_sametype_pipelinestage.phases[0].current_executors)
                upcoming_throughputs[str(pipelinestageindex)] += max_throughputs[str(pipelinestageindex)]
            else:
                upcoming_throughputs[str(pipelinestageindex)] = max_throughputs[str(pipelinestageindex)]

            pipelinestageindex -= 1

        return throughputs, pending_workloads, computation_pressures, available_resources, max_throughputs, total_cpu_input_pressure, total_gpu_input_pressure, total_cpu_throughput, total_gpu_throughput, upcoming_throughputs

    def calculate_pipeline_stats_new (self, rmanager, current_time, free_cpus, free_gpus):
        throughputs = {}
        pending_workloads = {}
        computation_pressures= {}
        available_resources = {}
        max_throughputs = {}
        upcoming_throughputs = {}

        total_cpu_input_pressure = 0
        total_gpu_input_pressure = 0
        total_cpu_throughput = 0
        total_gpu_throughput = 0

        pipelinestageindex = 0
        current_free_cpus = copy.deepcopy(free_cpus)
        current_free_gpus = copy.deepcopy(free_gpus)



        while pipelinestageindex < len (self.pipelinestages):
            pipelinestage = self.pipelinestages[pipelinestageindex]

            if pipelinestage.resourcetype == 'CPU':
                free_resources = current_free_cpus
            else:
                free_resources = current_free_gpus

            available_resources[str(pipelinestageindex)] = copy.deepcopy(pipelinestage.phases[0].current_executors)

            for free_resource_id in free_resources:
                free_resource = rmanager.get_resource (free_resource_id, True)

                if free_resource.active == False:
                    if free_resource.temporary_assignment == str (pipelinestageindex):
                        available_resources[str(pipelinestageindex)].append (free_resource_id)
                        free_resources.remove (free_resource_id)


            throughputs[str(pipelinestageindex)] = pipelinestage.get_current_throughput(0)

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
                throughputs[str(pipelinestageindex - 1)] = prev_pipelinestage.get_current_throughput(0)
                computation_pressures[str(pipelinestageindex)] = [throughputs[str(pipelinestageindex - 1)],
                                                                  throughputs[str(pipelinestageindex)]]

            pipelinestageindex += 1


        pipelinestageindex = len (self.pipelinestages) - 1


        while pipelinestageindex >= 0:

            pipelinestage = self.pipelinestages[pipelinestageindex]

            if pipelinestageindex - 2 >= 0:
                prev_sametype_pipelinestage = self.pipelinestages[pipelinestageindex - 2]
            else:
                prev_sametype_pipelinestage = None

            if pipelinestage.resourcetype == 'CPU':
                free_resources = current_free_cpus
            else:
                free_resources = current_free_gpus


            if throughputs[str(pipelinestageindex)] <= 0:
                if computation_pressures[str(pipelinestageindex)][0] <= 0:
                    #available_resources[str(pipelinestageindex)] = []
                    max_throughputs[str(pipelinestageindex)] = 0
                else:
                    available_resources[str(pipelinestageindex)].extend (copy.deepcopy(free_resources))
                    max_throughputs[str(pipelinestageindex)] = pipelinestage.get_free_resource_throughput(rmanager,
                                                                                                          available_resources[str (pipelinestageindex)])
                    free_resources.clear()
            else:
                available_resources[str(pipelinestageindex)].extend (free_resources)
                max_throughputs[str(pipelinestageindex)] = pipelinestage.get_free_resource_throughput(rmanager,
                                                                                                      available_resources[str(pipelinestageindex)])
                free_resources.clear()

            if prev_sametype_pipelinestage != None:
                upcoming_throughputs[str(pipelinestageindex)] = pipelinestage.get_free_resource_throughput(rmanager, available_resources[str (pipelinestageindex - 2)])
                upcoming_throughputs[str(pipelinestageindex)] += max_throughputs[str(pipelinestageindex)]
            else:
                upcoming_throughputs[str(pipelinestageindex)] = max_throughputs[str(pipelinestageindex)]

            pipelinestageindex -= 1

        return throughputs, pending_workloads, computation_pressures, available_resources, max_throughputs, total_cpu_input_pressure, total_gpu_input_pressure, total_cpu_throughput, total_gpu_throughput, upcoming_throughputs

    def reconfiguration_up_down_underallocations (self, rmanager, current_time, free_cpus, free_gpus, imbalance_limit, throughput_target):
        print('reconfiguration_up_down_underallocations ()', free_cpus, free_gpus)

        throughputs, pending_workloads, computation_pressures, available_resources, max_throughputs, total_cpu_input_pressure, total_gpu_input_pressure, total_cpu_throughput, total_gpu_throughput, upcoming_throughputs = self.calculate_pipeline_stats_new(rmanager, current_time, free_cpus, free_gpus)

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

            throughput_limit = 0

            if pipelinestageindex < len (self.pipelinestages) - 1:
                next_pipelinestage_upcoming_throughput = upcoming_throughputs[str (pipelinestageindex + 1)]

                prev_pipelinestage_throughput = computation_pressures[str(pipelinestageindex)][0]

                if next_pipelinestage_upcoming_throughput < prev_pipelinestage_throughput:
                    throughput_limit = next_pipelinestage_upcoming_throughput
                else:
                    throughput_limit = prev_pipelinestage_throughput


            if str(pipelinestageindex) in underallocations and pending_workitems > 0 and ((underallocations[str(pipelinestageindex)] >= 1.0 and total_throughput <= 0) or underallocations[str(pipelinestageindex)] > 0 and throughputs[str(pipelinestageindex)] == total_throughput):
                print('reconfiguration_up_down_underallocations 1 ()', pipelinestage.name, underallocations,
                      computation_pressures[str(pipelinestageindex)], max_throughputs[str(pipelinestageindex)],
                      available_resources, pending_workloads)

                to_be_added = self.scale_up_configuration_limit(rmanager, pipelinestageindex,
                                                                computation_pressures[str(pipelinestageindex)][0],
                                                                max_throughputs[str(pipelinestageindex)], pending_workitems, throughput_limit)

                #to_be_added = self.scale_up_configuration_limit_imbalance_limit(rmanager, pipelinestageindex,
                #                                                computation_pressures[str(pipelinestageindex)][0],
                #                                                max_throughputs[str(pipelinestageindex)],
                #                                                pending_workitems, imbalance_limit)
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
                                                                pending_workitems, throughput_limit)

                #to_be_added = self.scale_up_configuration_limit_imbalance_limit(rmanager, pipelinestageindex,
                #                                                computation_pressures[str(pipelinestageindex)][0],
                #                                                max_throughputs[str(pipelinestageindex)],
                #                                                pending_workitems, imbalance_limit)
                if pipelinestage.resourcetype == 'CPU':
                    print('CPUs to be added', to_be_added)
                    cpus_to_be_added[str(pipelinestageindex)] = to_be_added
                else:
                    print('GPUs to be added', to_be_added)
                    gpus_to_be_added[str(pipelinestageindex)] = to_be_added

            pipelinestageindex += 1

        return cpus_to_be_added, gpus_to_be_added

    def reconfiguration_up_down_overallocations (self, rmanager, current_time, free_cpus, free_gpus, imbalance_limit, throughput_target):

        print('reconfiguration_up_down_overallocations ()', free_cpus, free_gpus)

        throughputs, pending_workloads, computation_pressures, available_resources, max_throughputs, total_cpu_input_pressure, total_gpu_input_pressure, total_cpu_throughput, total_gpu_throughput, upcoming_throughputs = self.calculate_pipeline_stats_new(rmanager, current_time, free_cpus, free_gpus)

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

                #to_be_dropped = self.scale_down_configuration_imbalance_limit(rmanager, pipelinestageindex,
                #                                              overallocations[str(pipelinestageindex)],
                #                                              max_throughputs[str(pipelinestageindex)],
                #                                              available_resources[str(pipelinestageindex)],
                #                                                              imbalance_limit)

                if pipelinestage.resourcetype == 'CPU':
                    print('CPUs to be dropped', to_be_dropped)
                    cpus_to_be_dropped.extend(to_be_dropped)
                else:
                    print('GPUs to be dropped', to_be_dropped)
                    gpus_to_be_dropped.extend(to_be_dropped)

            pipelinestageindex += 1

        return cpus_to_be_dropped, gpus_to_be_dropped

    def reconfiguration_drop (self, rmanager, current_time, free_cpus, free_gpus, imbalance_limit, throughput_target):

        gpus_to_be_dropped = []
        cpus_to_be_dropped = []

        print ('reconfiguration_drop ()', free_cpus, free_gpus)

        throughputs, pending_workloads, computation_pressures, available_resources, max_throughputs, total_cpu_input_pressure, total_gpu_input_pressure, total_cpu_throughput, total_gpu_throughput, upcoming_throughputs = self.calculate_pipeline_stats_new(rmanager, current_time, free_cpus, free_gpus)

        if total_gpu_throughput <= 0 and total_gpu_input_pressure <= 0:
            gpu_weighted_pcr_ranking = self.get_weighted_performance_to_cost_ratio_ranking(rmanager, 'GPU', free_gpus)

            to_be_dropped = []

            for gpu_id in gpu_weighted_pcr_ranking.keys():
                gpu = rmanager.get_resource(gpu_id, active=True)
                to_be_dropped.append(gpu.id)
            print('GPUs to be dropped', to_be_dropped)
            gpus_to_be_dropped.extend(to_be_dropped)

        if total_cpu_throughput <= 0 and total_cpu_input_pressure <= 0:
            cpu_weighted_pcr_ranking = self.get_weighted_performance_to_cost_ratio_ranking(rmanager, 'CPU', free_cpus)

            to_be_dropped = []

            for cpu_id in cpu_weighted_pcr_ranking.keys():
                cpu = rmanager.get_resource(cpu_id, active=True)
                to_be_dropped.append(cpu.id)
            print('CPUs to be dropped', to_be_dropped)
            cpus_to_be_dropped.extend(to_be_dropped)

        print('total throughput', total_cpu_throughput, total_gpu_throughput)
        print('total input pressure', total_cpu_input_pressure, total_gpu_input_pressure)
        print('computation pressures', computation_pressures)
        print('max throughputs', max_throughputs)
        print('available resources', available_resources)
        print ('pending_workloads ', pending_workloads)

        return cpus_to_be_dropped, gpus_to_be_dropped


    def parse_pipelines (self, rmanager):
        pipelinedatafile = open(self.pipelinefile)
        pipelinedata = yaml.load(pipelinedatafile, Loader=yaml.FullLoader)

        index = 0
        for pipelinestage_node in pipelinedata['pipelinestages']:
            pipelinestage_name = pipelinestage_node['name']
            pipelinestage_resourcetype = pipelinestage_node['resourcetype']

            new_pipelinestage = PipelineStage(index, pipelinestage_name, pipelinestage_resourcetype, rmanager, self.batchsize)

            self.pipelinestages.append(new_pipelinestage)
            self.pipelinestages_dict[pipelinestage_name] = new_pipelinestage

            index += 1

        for pipelinestage_node in pipelinedata['pipelinestages']:
            pipelinestage = self.pipelinestages_dict[pipelinestage_node['name']]
            if 'exec_dependencies' in pipelinestage_node:
                exec_dependencies = pipelinestage_node['exec_dependencies']

                for dependency in exec_dependencies:
                    parent_pipelinestage = self.pipelinestages_dict[dependency]
                    pipelinestage.add_dependency_parent (parent_pipelinestage, 'exec')
                    parent_pipelinestage.add_dependency_child (pipelinestage, 'exec')

            if 'data_dependencies' in pipelinestage_node:
                data_dependencies = pipelinestage_node['data_dependencies']

                for dependency in data_dependencies:
                    parent_pipelinestage = self.pipelinestages_dict[dependency]
                    pipelinestage.add_dependency_parent (parent_pipelinestage, 'data')
                    parent_pipelinestage.add_dependency_child (pipelinestage, 'data')

        for pipelinestage in self.pipelinestages:
            pipelinestage.print_data ()

    def parse_pipelines_old (self, rmanager):
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


    def close_phases_fixed (self, rmanager, anyway):

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


    def add_executor (self, workitem, resource, now):
        #print ('add_executor ()', workitem.id, workitem.version,  resource.id)
        pipelinestage = self.pipelinestages[int (workitem.version)]
        pipelinestage.add_executor (workitem, resource, now)
        #if int (workitem.version) > 0:
        #    prev_pipelinestage = self.pipelinestages[int (workitem.version) - 1]
        #    prev_pipelinestage.remove_output (workitem)

    def remove_executor (self, workitem, resource, now):
        #print('remove_executor ()', workitem.id, workitem.version, resource.id)
        pipelinestage = self.pipelinestages[int (workitem.version)]
        #pipelinestage.add_output(workitem)
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

    def remove_workitem_queue (self, workitem, current_time):
        if workitem == None:
            print ('invalid workitem')
            return

        if workitem.iscomplete == True:
            if int (workitem.version)  <= len (self.pipelinestages) - 1:
                pipelinestage_remove = self.pipelinestages[int (workitem.version)]
            else:
                pipelinestage_remove = None

            if pipelinestage_remove != None:
                remove_phase, remove_phase_index = pipelinestage_remove.get_phase (workitem)

                if remove_phase != None:
                    remove_phase.remove_workitem (current_time, workitem)

    def add_workitem_queue (self, add_workitem, current_time, new_workitem_index = -1):
        pipelinestage_remove = None
        pipelinestage_add = None

        if add_workitem == None:
            print ('invalid workitem')
            return
        #print ('add_workitem_queue', workitem.id, workitem.iscomplete)

        pipelinestage_add = self.pipelinestages[int(add_workitem.version)]
        if pipelinestage_add.index == 0:
            if new_workitem_index == -1:
                pipelinestage_add.add_new_workitem(add_workitem, current_time, self.last_first_phase_closed_index)
            else:
                pipelinestage_add.add_new_workitem(add_workitem, current_time, new_workitem_index - 1)
        elif int (add_workitem.version) < len (self.pipelinestages):
            pipelinestage_add = self.pipelinestages[int(add_workitem.version)]

            add_phase, add_phase_index = pipelinestage_add.get_phase(add_workitem)

            add_phase.add_workitem(add_workitem, current_time)

    def check_new_workitem_index (self):
        first_pipelinestage = self.pipelinestages[0]

        #print ('check_new_workitem_index ()', self.last_first_phase_closed_index, first_pipelinestage.phases[self.last_first_phase_closed_index + 1].total_count)
        if first_pipelinestage.phases[self.last_first_phase_closed_index + 1].total_count == self.batchsize:
            return self.last_first_phase_closed_index + 2

        return self.last_first_phase_closed_index + 1

    def build_phases (self):
        for i in range(0, self.no_of_columns):
            index = 0
            while index < len (self.pipelinestages):
                current_stage = self.pipelinestages[index]
                current_stage.create_phase ()
                index += 1

    def print_stage_queue_data_3 (self, actual_idle_periods):
        plot_prediction_idle_periods (actual_idle_periods, self.prediction_idle_periods)

    def print_stage_queue_data_2 (self, rmanager):
        plot_data = {}
        for pipelinestage in self.pipelinestages:
            plot_data[pipelinestage.name] = []
            for phase in pipelinestage.phases:
                plot_data[pipelinestage.name].append ([phase.queue_snapshots, phase.starttime, phase.endtime, phase.predictions])

        plot_prediction_sim_0 (self, rmanager, plot_data, self.prediction_times, self.batchsize)


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

    def get_first_pipelinestage (self):
        return self.pipelinestages[0]

    def get_next_pipelinestages (self, current_pipelinestage):
        children = current_pipelinestage.get_children ('exec')

        return children

    def get_pipelinestage (self, index):
        return self.pipelinestages[index]

    '''
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

        return ret
    '''

if __name__ == "__main__":
    pipelinefile = sys.argv[1]
    p = PipelineManager(pipelinefile)
    p.parse_pipelines ()
    p.print_data ()
