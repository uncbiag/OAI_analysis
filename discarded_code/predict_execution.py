def predict_execution(self, rmanager, pmanager, current_time):
    cpu_resources = rmanager.get_resources_type('CPU')
    gpu_resources = rmanager.get_resources_type('GPU')

    if self.added_new_phases == False:
        return

    self.added_new_phases = False

    if len(self.pipelinestages[0].phases) <= 2:
        return

    index = len(self.pipelinestages[0].phases) - 2

    prev_column_phase = None
    prev_prev_column_phase = None

    while index < len(self.pipelinestages[0].phases):

        prev_prev_pipelinestage_phase = None
        prev_pipelinestage_phase = None
        prev_pipelinestage = None
        prev_prev_pipelinestage = None

        pipelinestage_index = 0
        for pipelinestage in self.pipelinestages:

            phase = pipelinestage.phases[index]

            if phase.active == False and phase.complete == True:  # phase complete
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

                if prev_column_phase == None:  # first index
                    next_pipelinestage = self.pipelinestages[pipelinestage_index + 1]
                    next_next_pipelinestage = self.pipelinestages[pipelinestage_index + 2]
                    next_pipelinestage_phase = next_pipelinestage.phases[index]
                    next_next_pipelinestage_phase = next_next_pipelinestage.phases[index]

                    phase_executors_count = len(phase.current_executors)

                    next_next_pipelinestage_phase_pending_count = next_next_pipelinestage_phase.current_count - \
                                                                  len(next_next_pipelinestage_phase.current_executors)

                    if phase_executors_count <= next_next_pipelinestage_phase_pending_count:
                        phase.ptarget = phase.total_count
                        phase.pendtime = current_time + queued_work / executors_service_rate
                        phase.pending_output = queued_work
                    else:
                        target = phase_executors_count - next_next_pipelinestage_phase_pending_count
                        time_left = next_pipelinestage_phase.get_time_required(rmanager, self,
                                                                               pipelinestage_index + 1, index, target,
                                                                               pipelinestage.resourcetype, current_time)

                        print('predict_execution', 'time_left', phase_executors_count,
                              next_next_pipelinestage_phase_pending_count, target, time_left)
                        possible_completions = phase.possible_completions(rmanager, pipelinestage.resourcetype,
                                                                          current_time, time_left)
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
                        phase.first_resource_release_target = phase.ptarget - (len(pipelinestage_resources) - 1)
                        phase.first_resource_release_time = phase.starttime + (
                                    phase.first_resource_release_target / all_resource_service_rate)

                    phase.predictions.append(
                        [current_time, index, pipelinestage_index, pipelinestage.name,
                         phase.starttime, phase.pstarttime, phase.ptarget, phase.pendtime,
                         phase.pending_output, phase.first_output_completion_time, phase.first_resource_release_time])
                else:  # second and beyond index
                    pass
            else:

                if prev_column_phase == None:  # first index

                    if prev_prev_pipelinestage_phase == None:
                        prev_sametype_phase = None
                    else:
                        prev_sametype_phase = prev_prev_pipelinestage_phase

                    if phase.starttime == -1:
                        if prev_sametype_phase == None:
                            print('oops 1')
                        else:
                            if prev_pipelinestage_phase.first_output_completion_time > prev_sametype_phase.first_resource_release_time:
                                phase.pstarttime = prev_pipelinestage_phase.first_output_completion_time
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
                            phase.pendtime = current_time + phase.pendtime_output / phase.executors_service_rate  # include incoming executors if any

                        work_till_release = phase.pending_output - (len(pipelinestage_resources) - 1)

                        if phase.starttime == -1:
                            phase.first_output_completion_time = phase.pstarttime + 1 / avg_resource_service_rate  # use avg. all resource_service_rate
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
                        phase.pendtime = prev_pipelinestage_phase.pendtime + len(
                            pipelinestage_resources) / all_resource_service_rate

                        if phase.starttime == -1:
                            phase.first_output_completion_time = phase.pstarttime + 1 / avg_resource_service_rate  # use avg. all resource_service_rate
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
                output += " " + str(item)

            print(output)

            prev_prev_pipelinestage_phase = prev_pipelinestage_phase
            prev_pipelinestage_phase = phase
            pipelinestage_index += 1

        prev_column_phase = prev_pipelinestage_phase
        prev_prev_column_phase = prev_prev_pipelinestage_phase
        index += 1