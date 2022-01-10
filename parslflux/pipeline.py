import yaml
import sys
from parslfluxsim.resources_sim import Resource
import statistics

class Phase:
    def __init__(self, pipelinestage, index, resourcetype, obj):
        self.pipelinestage = pipelinestage
        self.pipelinestage_object = obj
        self.resourcetype = resourcetype
        self.index = index
        self.ptarget = -1
        self.starttime = -1
        self.current_count = 0
        self.total_count = 0
        self.total_complete = 0
        self.add_timestamps = {}
        self.remove_timestamps = {}
        self.endtime = -1
        self.active = False
        self.complete = False
        self.current_executors = []
        self.workitems = []
        self.pstarttime = -1
        self.pendtime = -1
        self.predictions = []

    def get_completed_count (self):
        return self.total_count - self.current_count

    def get_executors (self):
        return self.current_executors

    def add_executor (self, resource, now):
        if resource.id not in self.current_executors:
            self.current_executors.append(resource.id)
            if self.total_complete == 0 and self.starttime == -1:
                self.starttime = now
                self.pstarttime = now

    def remove_executor (self, resource):
        if resource.id in self.current_executors:
            self.current_executors.remove(resource.id)

    def add_workitem (self, workitem, currenttime):
        if self.active == False:
            self.active = True
            #print(self.pipelinestage, 'activate phase')
        self.current_count += 1
        self.total_count += 1
        self.add_timestamps[workitem.id + ':' + str (currenttime)] = self.current_count
        self.workitems.append(workitem.id)
        #print(self.pipelinestage, 'add workitem', self.index, currenttime, workitem.id, self.current_count)

    def remove_workitem (self, currenttime, workitem):
        self.current_count -= 1
        self.remove_timestamps[workitem.id + ':' + str (currenttime)] = self.current_count
        self.workitems.remove (workitem.id)
        self.total_complete += 1
        if self.total_complete == 1:
            self.first_workitem_completion_time = currenttime
        #print(self.pipelinestage, 'remove workitem', self.index, currenttime, workitem.id, self.current_count)

    def close_phase (self):
        self.active = False
        self.complete = True

        timestamps_list = list (self.remove_timestamps.keys())
        if len (timestamps_list) < 5:
            first_resource_release_timestamp = timestamps_list[0]
        else:
            first_resource_release_timestamp = timestamps_list[-5]
        self.first_resource_release_time = first_resource_release_timestamp.split (':')[1]
        self.endtime = timestamps_list[-1].split (':')[1]

        print('phase closed', self.pipelinestage, self.starttime, self.first_workitem_completion_time,
              self.first_resource_release_time, self.endtime, self.total_count, float(self.endtime) - float(self.starttime),
              float(self.endtime) - float(self.first_resource_release_time))

    def get_queued_work (self, rmanager, resourcetype, current_time):
        queued_work = self.current_count - len(self.current_executors)

        for executor in self.current_executors:
            executor = rmanager.get_resource(executor)
            work_left = executor.get_work_left(self, resourcetype, current_time)
            if work_left == None:
                print('workitem doesnt exist')
                continue

            queued_work += work_left

        self.queued_work = queued_work

        return self.queued_work

    def get_executors_service_rate (self, rmanager):
        executors_service_rates = []
        for executor in self.current_executors:
            executor = rmanager.get_resource (executor)
            exectime = executor.get_exectime(self.pipelinestage)
            if exectime == 0:
                print('exectime does not exist')
                continue

            executors_service_rates.append(1 / exectime)

        self.executors_service_rate = sum(executors_service_rates)

        return self.executors_service_rate

    def get_possible_completions (self, rmanager, resourcetype, current_time, time_left):
        queued_work = self.get_queued_work(rmanager, resourcetype, current_time)
        queued_work_time_required = queued_work / self.executors_service_rate

        if queued_work_time_required >= time_left:
            return 0
        else:
            return (time_left - queued_work_time_required) * self.executors_service_rate

    def get_prev_phase (self, pmanager, pipelinestage_index, phase_index):
        if pipelinestage_index <= 0:
            return None

        prev_pipelinestage = pmanager.pipelinestages[pipelinestage_index - 1]
        prev_phase = prev_pipelinestage.phases[phase_index]

        return prev_phase

    def get_prev_sametype_phase (self, pmanager, pipelinestage_index, phase_index):
        current_pipelinestage = pmanager.pipelinestages[pipelinestage_index]

        if pipelinestage_index <= 1:
            prev_index_last_pipelinestage = pmanager.pipelinestages[-1]
            if prev_index_last_pipelinestage.resourcetype == current_pipelinestage:
                return prev_index_last_pipelinestage.phases[phase_index - 1]
            else:
                prev_index_last_but_one_pipelinestage = pmanager.pipelinestages[-2]
                return prev_index_last_but_one_pipelinestage.phases[phase_index - 1]
        else:
            prev_prev_pipelinestage = pmanager.pipelinestages[pipelinestage_index - 1]
            return prev_prev_pipelinestage.phases[phase_index]

    def get_time_required (self, rmanager, pmanager, pipelinestage_index, phase_index, target, resourcetype, current_time):
        queued_work = self.get_queued_work(rmanager, resourcetype, current_time)

        #print('get_time_required ()', self.pipelinestage, pipelinestage_index, phase_index, queued_work, target)

        executors_service_rate = self.get_executors_service_rate(rmanager)
        all_resource_service_rate = self.pipelinestage_object.get_resource_service_rate (rmanager)

        prev_phase = self.get_prev_phase(pmanager, pipelinestage_index, phase_index)

        if queued_work >= target:
            time_required = target / executors_service_rate
        else:
            prev_phase_target = target - queued_work

            if prev_phase != None:
                prev_phase_time_required = prev_phase.get_time_required (rmanager, pmanager, pipelinestage_index - 1,
                                                                         phase_index, prev_phase_target,
                                                                         pmanager.pipelinestages[pipelinestage_index - 1].resourcetype,
                                                                         current_time)

                time_required = prev_phase_time_required + prev_phase_target
            else:
                if prev_phase_target > 0:
                    print ('oops 0')
                prev_phase_time_required = 0

            if self.starttime == -1:
                if self.pstarttime > current_time:
                    time_required = (self.pstarttime - current_time) + prev_phase_time_required + queued_work / all_resource_service_rate
                else:
                    time_required = prev_phase_time_required + queued_work / all_resource_service_rate
            else:
                if executors_service_rate > 0:
                    time_required = prev_phase_time_required + queued_work / executors_service_rate
                else:
                    time_required = prev_phase_time_required + queued_work / all_resource_service_rate

        #print ('get_time_required ()', time_required)

        return time_required

    def print_data (self):
        print('print_data ()', self.pipelinestage, self.starttime, self.first_workitem_completion_time,
              self.first_resource_release_time, self.endtime, self.total_count,
              float(self.endtime) - float(self.starttime),
              float(self.endtime) - float(self.first_resource_release_time)
              )
        #print (self.timestamps)


class PipelineStage:
    def __init__ (self, stageindex, names, resourcetype):
        index = 0
        self.name = names[index]
        index += 1
        while index < len (names):
            self.name = self.name + ":" + names[index]
            index += 1
        self.index = stageindex
        self.resourcetype = resourcetype
        self.phases = []

    def create_phase (self):
        print(self.name, 'create phase')

        new_phase = Phase(self.name, len (self.phases), self.resourcetype, self)
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
        index = 0
        for phase in self.phases:
            #print (phase.workitems)
            if phase.active == False:
                index += 1
                continue
            if workitem.id in phase.workitems:
                return phase, index
            index += 1
        return None, None

    def get_phase_index (self, index):
        #print('get_phase_index', len(self.phases), index)
        if index <= (len(self.phases) - 1):
            return self.phases[index]

        return None

    def add_workitem_index (self, index, workitem, current_time):

        phase = self.phases[index]

        phase.add_workitem (workitem, current_time)

        #print(self.name, 'add workitem index', workitem.id, len(self.phases) - 1, phase.current_count)

    def add_new_workitem (self, workitem, current_time):
        latest_phase = self.phases[-2]

        latest_phase.add_workitem(workitem, current_time)
        workitem.phase_index = len (self.phases) - 2
        #print (self.name, 'add workitem', workitem.id, len (self.phases) - 1, latest_phase.current_count)

    def add_executor (self, workitem, resource, now):
        current_phase, index = self.get_phase(workitem)
        if current_phase == None:
            print ('add_executor', workitem.id, 'not found')
            return
        current_phase.add_executor (resource, now)
        #print(self.name, 'add executor', now, workitem.id, len(self.phases) - 1, index)

    def remove_executor (self, workitem, resource):
        current_phase, index = self.get_phase(workitem)
        if current_phase == None:
            print('remove_executor', workitem.id, self.name, 'not found')
            return
        current_phase.remove_executor (resource)
        #print(self.name, 'remove executor', workitem.id, len(self.phases) - 1, index)

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

    def print_data (self, index):
        if index < len (self.phases):
            self.phases[index].print_data ()

class PipelineManager:
    def __init__ (self, pipelinefile, budget, batchsize):
        self.pipelinefile = pipelinefile
        self.pipelinestages = []
        self.added_new_phases = False
        self.batchsize = batchsize
        self.budget = budget

    def parse_pipelines (self):
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
                self.pipelinestages.append (PipelineStage(current_index, current_names, current_resourcetype))
                current_index = index
                current_names.clear()
                current_resourcetype = pipelinestage_resourcetypes[index]
                current_names.append(pipelinestage_names[index])
            index += 1
        self.pipelinestages.append(PipelineStage(current_index, current_names, current_resourcetype))

    def predict_execution (self, rmanager, pmanager, current_time):
        cpu_resources = rmanager.get_resources_type ('CPU')
        gpu_resources = rmanager.get_resources_type ('GPU')

        if self.added_new_phases == False:
            return

        self.added_new_phases = False

        if len (self.pipelinestages[0].phases) <= 2:
            return

        index = len (self.pipelinestages[0].phases) - 2

        prev_column_phase = None
        prev_prev_column_phase = None

        while index < len (self.pipelinestages[0].phases):

            prev_prev_pipelinestage_phase = None
            prev_pipelinestage_phase = None
            prev_pipelinestage = None
            prev_prev_pipelinestage = None

            pipelinestage_index = 0
            for pipelinestage in self.pipelinestages:

                phase = pipelinestage.phases[index]

                if phase.active == False and phase.complete == True: #phase complete
                    pipelinestage_index += 1
                    continue

                # Calculate queued work

                queued_work = phase.get_queued_work(rmanager, pipelinestage.resourcetype, current_time)

                # calculate service rates
                all_resource_service_rate = pipelinestage.get_resource_service_rate(rmanager)
                avg_resource_service_rate = pipelinestage.get_avg_resource_service_rate(rmanager)
                executors_service_rate = phase.get_executors_service_rate(rmanager)
                pipelinestage_resources = rmanager.get_resources_type(pipelinestage.resourcetype)

                if prev_pipelinestage_phase == None:

                    if prev_column_phase == None: #first index
                        next_pipelinestage = self.pipelinestages[pipelinestage_index + 1]
                        next_next_pipelinestage = self.pipelinestages[pipelinestage_index + 2]
                        next_pipelinestage_phase = next_pipelinestage.phases[index]
                        next_next_pipelinestage_phase = next_next_pipelinestage.phases[index]

                        phase_executors_count = len (phase.current_executors)

                        next_next_pipelinestage_phase_pending_count = next_next_pipelinestage_phase.current_count -\
                                                                      len (next_next_pipelinestage_phase.current_executors)

                        if phase_executors_count <= next_next_pipelinestage_phase_pending_count:
                            phase.ptarget = phase.total_count
                            phase.pendtime = current_time + queued_work / executors_service_rate
                            phase.pending_output = queued_work
                        else:
                            target = phase_executors_count - next_next_pipelinestage_phase_pending_count
                            time_left = next_pipelinestage_phase.get_time_required (rmanager, self,
                                                                                    pipelinestage_index + 1, index, target,
                                                                                    pipelinestage.resourcetype, current_time)

                            print ('predict_execution', 'time_left', phase_executors_count, next_next_pipelinestage_phase_pending_count, target, time_left)
                            possible_completions = phase.possible_completions(rmanager, pipelinestage.resourcetype, current_time, time_left)
                            phase.ptarget = phase.total_count + possible_completions
                            phase.pendtime = current_time + (queued_work + possible_completions) / executors_service_rate
                            phase.pending_output = queued_work + possible_completions

                        if phase.pstarttime == -1:
                            phase.first_output_completion_time = phase.pstarttime + 1 / avg_resource_service_rate
                            phase.first_resource_release_target = phase.ptarget - (len(pipelinestage_resources) - 1)
                            phase.first_resource_release_time = phase.pstarttime + (
                                        phase.first_resource_release_target / all_resource_service_rate)
                        else:
                            phase.first_output_completion_time = phase.starttime + 1 / avg_resource_service_rate

                            pipelinestage_resources = rmanager.get_resources_type(pipelinestage.resourcetype)
                            phase.first_resource_release_target = phase.ptarget - (len (pipelinestage_resources) - 1)
                            phase.first_resource_release_time = phase.starttime + (phase.first_resource_release_target / all_resource_service_rate)

                        phase.predictions.append(
                            [current_time, index, pipelinestage_index, pipelinestage.name,
                             phase.starttime, phase.pstarttime, phase.ptarget, phase.pendtime,
                             phase.pending_output, phase.first_output_completion_time, phase.first_resource_release_time])
                    else: #second and beyond index
                        pass
                else:

                    if prev_column_phase == None:#first index

                        if prev_prev_pipelinestage_phase == None:
                            prev_sametype_phase = None
                        else:
                            prev_sametype_phase = prev_prev_pipelinestage_phase

                        if phase.starttime == -1:
                            if prev_sametype_phase == None:
                                print ('oops 1')
                            else:
                                if prev_pipelinestage_phase.first_output_completion_time > prev_sametype_phase.first_resource_release_time:
                                    phase.pstarttime =  prev_pipelinestage_phase.first_output_completion_time
                                else:
                                    phase.pstarttime = prev_sametype_phase.first_resource_release_time

                        phase.pending_output = queued_work + prev_pipelinestage_phase.pending_output

                        if prev_pipelinestage_phase.all_resource_service_rate > phase.all_resource_service_rate:
                            if prev_sametype_phase != None:
                                time_left = prev_sametype_phase.pendtime - current_time
                                if time_left > 0:
                                    work_done = executors_service_rate * time_left
                                    pending_work = phase.pending_output - work_done
                                    phase.pendtime = current_time + time_left + pending_work / all_resource_service_rate
                                else:
                                    phase.pendtime = phase.pstarttime + phase.pending_output / all_resource_service_rate
                            else:
                                phase.pendtime = current_time + phase.pendtime_output / phase.executors_service_rate #include incoming executors if any

                            work_till_release = phase.pending_output - (len(pipelinestage_resources) - 1)

                            if phase.starttime == -1:
                                phase.first_output_completion_time = phase.pstarttime + 1 / avg_resource_service_rate #use avg. all resource_service_rate
                                if phase.pending_output > len(pipelinestage_resources):
                                    phase.first_resource_release_time = phase.pstarttime + work_till_release / all_resource_service_rate
                            else:
                                phase.first_output_completion_time = phase.starttime + 1 / avg_resource_service_rate
                                if phase.pending_output > len(pipelinestage_resources):
                                    phase.first_resource_release_time = phase.starttime + work_till_release / all_resource_service_rate

                            phase.predictions.append(
                                [current_time, index, pipelinestage_index, pipelinestage.name,
                                 phase.starttime, phase.pstarttime, phase.pendtime,
                                 phase.pending_output, phase.first_output_completion_time,
                                 phase.first_resource_release_time])
                        else:
                            phase.pendtime = prev_pipelinestage_phase.pendtime + len (pipelinestage_resources) / all_resource_service_rate

                            if phase.starttime == -1:
                                phase.first_output_completion_time = phase.pstarttime + 1 / avg_resource_service_rate #use avg. all resource_service_rate
                                if phase.pending_output > len(pipelinestage_resources):
                                    phase.first_resource_release_time = prev_pipelinestage_phase.pendtime + 1 / avg_resource_service_rate
                            else:
                                phase.first_output_completion_time = phase.starttime + 1 / avg_resource_service_rate
                                if phase.pending_output > len(pipelinestage_resources):
                                    phase.first_resource_release_time = prev_pipelinestage_phase.pendtime + 1 / all_resource_service_rate

                            phase.predictions.append(
                                [current_time, index, pipelinestage_index, pipelinestage.name,
                                 phase.starttime, phase.pstarttime, phase.pendtime,
                                 phase.pending_output, phase.first_output_completion_time,
                                 phase.first_resource_release_time])

                prediction = phase.predictions[-1]

                output = ""
                for item in prediction:
                    output += " " + str (item)

                print (output)

                prev_prev_pipelinestage_phase = prev_pipelinestage_phase
                prev_pipelinestage_phase = phase
                pipelinestage_index += 1

            prev_column_phase = prev_pipelinestage_phase
            prev_prev_column_phase = prev_prev_pipelinestage_phase
            index += 1

    def predict_execution_fixed (self, rmanager, pmanager, current_time, batchsize, last_phase_index_closed):
        index = last_phase_index_closed + 1

        while index < last_phase_index_closed + 2:

            prev_prev_pipelinestage_phase = None
            prev_pipelinestage_phase = None
            prev_pipelinestage = None
            prev_prev_pipelinestage = None

            pipelinestage_index = 0
            for pipelinestage in self.pipelinestages:

                phase = pipelinestage.phases[index]

                if phase.active == False and phase.complete == True: #phase complete
                    pipelinestage_index += 1
                    continue

                # Calculate queued work
                queued_work = phase.get_queued_work(rmanager, pipelinestage.resourcetype, current_time)

                # calculate service rates
                all_resource_service_rate = pipelinestage.get_resource_service_rate(rmanager)
                avg_resource_service_rate = pipelinestage.get_avg_resource_service_rate(rmanager)
                executors_service_rate = phase.get_executors_service_rate(rmanager)
                pipelinestage_resources = rmanager.get_resources_type(pipelinestage.resourcetype)
                phase.target = batchsize

                if prev_pipelinestage_phase == None:
                    phase.pending_output = queued_work + (batchsize - phase.total_count)
                    phase.pendtime = current_time + phase.pending_output / all_resource_service_rate
                    phase.first_output_completion_time = phase.starttime + 1 / avg_resource_service_rate #use fastest

                    phase.first_resource_release_target = phase.pending_output - (len(pipelinestage_resources) - 1)
                    phase.first_resource_release_time = current_time + (
                                phase.first_resource_release_target / all_resource_service_rate)

                    phase.predictions.append(
                        [current_time, index, pipelinestage.name,
                         phase.starttime, phase.pstarttime, phase.pendtime,
                         phase.pending_output, phase.first_output_completion_time, phase.first_resource_release_time])
                else:
                    prev_sametype_phase = prev_prev_pipelinestage_phase

                    phase.pending_output = queued_work + prev_pipelinestage_phase.pending_output

                    if phase.starttime == -1:
                        if prev_sametype_phase == None:
                            phase.pstarttime = prev_pipelinestage_phase.first_output_completion_time
                        else:
                            if prev_pipelinestage_phase.first_output_completion_time > prev_sametype_phase.first_resource_release_time:
                                phase.pstarttime = prev_pipelinestage_phase.first_output_completion_time
                            else:
                                phase.pstarttime = prev_sametype_phase.first_resource_release_time

                        phase.first_output_completion_time = phase.pstarttime + 1 / avg_resource_service_rate
                    else:
                        phase.first_output_completion_time = phase.starttime + 1 / avg_resource_service_rate

                    work_till_release = phase.pending_output - (len(pipelinestage_resources) - 1)

                    if prev_pipelinestage.all_resource_service_rate > pipelinestage.all_resource_service_rate:
                        phase.pendtime = phase.pstarttime + phase.pending_output / all_resource_service_rate
                        if phase.starttime == -1:
                            if phase.pending_output > len(pipelinestage_resources):
                                phase.first_resource_release_time = phase.pstarttime + work_till_release / all_resource_service_rate
                        else:
                            if phase.pending_output > len(pipelinestage_resources):
                                phase.first_resource_release_time = phase.starttime + work_till_release / all_resource_service_rate

                        phase.predictions.append(
                            [current_time, index, pipelinestage.name,
                             phase.starttime, phase.pstarttime, phase.pendtime,
                             phase.pending_output, phase.first_output_completion_time,
                             phase.first_resource_release_time])
                    else:
                        phase.pendtime = prev_pipelinestage_phase.pendtime + len(
                            pipelinestage_resources) / all_resource_service_rate
                        if phase.starttime == -1:
                            if phase.pending_output > len(pipelinestage_resources):
                                phase.first_resource_release_time = prev_pipelinestage_phase.pendtime + 1 / avg_resource_service_rate
                        else:
                            if phase.pending_output > len(pipelinestage_resources):
                                phase.first_resource_release_time = prev_pipelinestage_phase.pendtime + 1 / avg_resource_service_rate

                        predicted_time_to_completion = phase.pendtime - phase.pstarttime
                        fastest_time_to_completion = batchsize / all_resource_service_rate

                        idle_period = predicted_time_to_completion - fastest_time_to_completion

                        phase.predictions.append(
                            [current_time, index, pipelinestage.name,
                             phase.starttime, phase.pstarttime, phase.pendtime,
                             phase.pending_output, phase.first_output_completion_time,
                             phase.first_resource_release_time, idle_period, (idle_period / (phase.pendtime - phase.pstarttime) * 100)])

                prediction = phase.predictions[-1]

                output = ""
                for item in prediction:
                    output += " " + str(item)

                print(output)

                prev_prev_pipelinestage = prev_pipelinestage
                prev_pipelinestage = pipelinestage
                prev_prev_pipelinestage_phase = prev_pipelinestage_phase
                prev_pipelinestage_phase = phase
                pipelinestage_index += 1

            index += 1


    def close_phases_fixed (self, rmanager, nowtime):
        cpu_resources = rmanager.get_resources_type('CPU')
        gpu_resources = rmanager.get_resources_type('GPU')

        index = 0

        length = len(self.pipelinestages[0].phases) - 1

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

                if phase.total_complete == self.batchsize:
                    phase.close_phase()
                    if current_index_pipelinestage_index == len (self.pipelinestages) - 1:
                        latest_last_phase_closed = index
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

    def remove_executor (self, workitem, resource):
        pipelinestage = self.pipelinestages[int (workitem.version)]
        pipelinestage.remove_executor (workitem, resource)

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
                        self.added_new_phases = True
            else:
                print ('add_workitem_queue', 'oops', pipelinestage_add.index)
            pipelinestage_add.add_new_workitem (workitem, current_time)

    def add_workitem_queue (self, workitem, current_time):
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
                    if (pipelinestage_add.phases[-2].active == False and pipelinestage_add.phases[-2].complete == True) or\
                        (pipelinestage_add.phases[-2].total_count == self.batchsize):
                        self.build_phases(1)
                        self.added_new_phases = True
            else:
                print ('add_workitem_queue', 'oops', pipelinestage_add.index)
            pipelinestage_add.add_new_workitem (workitem, current_time)

    def build_phases (self, count):
        for i in range(0, count):
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
        for index in range (0, len(self.pipelinestages[0].phases) - 1):
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
