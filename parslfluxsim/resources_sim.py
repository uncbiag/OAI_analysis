import yaml
import copy
import datetime
import random as rand

from parslflux.workqueue import WorkItemQueue
from aws_cloud_sim.costmodel import AWSCostModel
from filesystem.pfs import PFS

class ComputationUnit:
    def __init__ (self, name, cost):
        self.name = name
        self.cost = cost
        self.workqueue = WorkItemQueue ()
        self.busy = False
        self.last_completion_time = None
        self.idle_periods = []

    def reinit (self):
        self.workqueue = WorkItemQueue ()
        self.busy = False
        self.last_completion_time = None

    def get_cost (self):
        return self.cost

    def get_name (self):
        return self.name

    def is_busy (self):
        return self.busy

    def set_busy (self, busy):
        self.busy = busy

    def set_last_completion_time (self, time):
        self.last_completion_time = time

    def get_last_completion_time (self):
        return self.last_completion_time

    def clear_completion_time (self):
        self.last_completion_time = None

    def set_idle_start_time (self, time):
        self.idle_start_time = time

    def add_idle_period (self, now):
        if self.idle_start_time != None:
            self.idle_periods.append ([self.idle_start_time, now])
        #self.idle_periods.append ([self.last_completion_time, now])

    def report_idle_periods(self, starttime, endtime):
        ret = []
        for idle_period in self.idle_periods:
            if (idle_period[0] <= starttime and idle_period[1] > starttime) or \
                (idle_period[0] >= starttime and idle_period[0] <= endtime):
                ret.append(idle_period)

        return ret

class Resource:

    def __init__ (self, id, rmanager, provision_type, env, pipelinestageindex, domain_id):
        self.id = "c" + str(id)
        self.hostname = "c" + str (id)
        self.domain_id = str (domain_id)
        self.cpu = None
        self.gpu = None
        self.workqueue = WorkItemQueue ()
        self.busy = False
        self.exectimes = {}
        self.max_exectimes = {}
        self.transfertimes = {}
        self.counts = {}
        self.executionqueues = [-1, -1]
        self.rmanager = rmanager
        self.active = False
        self.provision_type = provision_type
        self.env = env
        self.acquisition_time = None
        self.pipelinestageindex = pipelinestageindex
        self.explored = False

    def print_data (self):
        print (self.id)

    def set_active (self, active):
        print ('set_active ()', self.domain_id, self.id, active)
        self.active = active
        self.acquisition_time = self.env.now

    def get_active (self):
        return self.active

    def set_explored (self, status):
        print ('set_explored ()', self.domain_id, self.id, status)
        self.explored = status

    def get_explored (self):
        return self.explored

    def add_cpu (self, cpu, cost, now):
        self.cpu = ComputationUnit (cpu, cost)

    def add_gpu (self, gpu, cost, now):
        self.gpu = ComputationUnit (gpu, cost)

    def get_cost (self, resourcetype):
        if resourcetype == 'CPU' and self.cpu != None:
            return self.cpu.get_cost ()
        elif resourcetype == 'GPU' and self.gpu != None:
            return self.gpu.get_cost ()
        else:
            return None

    def remove_cpu (self):
        self.cpu = None

    def remove_gpu (self):
        self.gpu = None

    def is_idle (self, resourcetype):
        if resourcetype == 'CPU':
            if self.cpu == None:
                return False
            elif self.cpu.is_busy() == True:
                return False
        else:
            if self.gpu == None:
                return False
            elif self.gpu.is_busy() == True:
                return False
        return True

    def is_idle_whole (self):
        if self.busy == True:
            return False
        return True

    def is_empty_whole (self):
        #print (self.workqueue.is_empty ())
        if self.workqueue.is_empty () == True:
            return True

        return False

    def is_empty (self, resourcetype):
        if resourcetype == 'CPU':
            if self.cpu == None:
                return False
            elif self.cpu.workqueue.is_empty() == True:
                return True
        else:
            if self.gpu == None:
                return False
            elif self.gpu.workqueue.is_empty() == True:
                return True

    def clear_completion_times (self):
        if self.cpu != None:
            self.cpu.clear_completion_time ()

        if self.gpu != None:
            self.gpu.clear_completion_time ()

    def set_idle_start_time (self, resourcetype, now):
        if resourcetype == 'CPU' and self.cpu != None:
            self.cpu.set_idle_start_time (now)

        if resourcetype == 'GPU' and self.gpu != None:
            self.gpu.set_idle_start_time (now)

    def add_idle_period (self, resourcetype, now):
        if resourcetype == 'CPU' and self.cpu != None:
            #if self.cpu.get_last_completion_time () != None:
            self.cpu.add_idle_period (now)

        if resourcetype == 'GPU' and self.gpu != None:
            #if self.gpu.get_last_completion_time () != None:
            self.gpu.add_idle_period (now)

    def report_idle_periods (self, starttime, endtime):
        cpu_idle_periods = None
        if self.cpu != None:
            cpu_idle_periods = self.cpu.report_idle_periods (starttime, endtime)

        gpu_idle_periods = None
        if self.gpu != None:
            gpu_idle_periods = self.gpu.report_idle_periods(starttime, endtime)

        return cpu_idle_periods, gpu_idle_periods

    def mark_exploration_workitem (self, resourcetype, status):
        if resourcetype == 'CPU' and self.cpu == None:
            print (self.id, 'CPU not available')
            return
        if resourcetype == 'GPU' and self.gpu == None:
            print (self.id, 'GPU not available')
            return

        if resourcetype == 'CPU' and self.cpu.workqueue.is_empty() == False:
            workitem = self.cpu.workqueue.get_workitem()
            workitem.mark_exploration_item (status)

        elif resourcetype == 'GPU' and self.gpu.workqueue.is_empty() == False:
            workitem = self.gpu.workqueue.get_workitem()
            workitem.mark_exploration_item (status)

    def schedule (self, rmanager, pmanager, resourcetype, thread_exec, env):
        if resourcetype == 'CPU' and self.cpu == None:
            print (self.id, 'CPU not available')
            return
        if resourcetype == 'GPU' and self.gpu == None:
            print (self.id, 'GPU not available')
            return

        if resourcetype == 'CPU' and self.cpu.workqueue.is_empty () == False:
            timeout = self.get_timeout_value (rmanager, pmanager, resourcetype)
            self.add_idle_period(resourcetype, env.now)
            cpu_workitem = self.cpu.workqueue.get_workitem ()
            cpu_workitem.submit (timeout, thread_exec, env)
            self.cpu.set_busy (True)
            self.cpu.set_last_completion_time (None)
            return

        elif resourcetype == 'GPU' and self.gpu.workqueue.is_empty () == False:
            timeout = self.get_timeout_value (rmanager, pmanager, resourcetype)
            self.add_idle_period(resourcetype, env.now)
            gpu_workitem = self.gpu.workqueue.get_workitem()
            gpu_workitem.submit(timeout, thread_exec, env)
            self.gpu.set_busy (True)
            self.gpu.set_last_completion_time (None)
            return

    def get_timeout_value_whole (self, rmanager, pmanager):
        return 20000

    def set_busy (self, busy):
        self.busy = busy

    def get_timeout_value (self, rmanager, pmanager, resourcetype):
        #print ('get_timeout_value ():', self.id)

        if resourcetype == 'CPU' and self.cpu != None and self.cpu.workqueue.is_empty () == False:
            workitem = self.cpu.workqueue.get_workitem ()
            workitem_pipelinestages = workitem.get_pipelinestage ()

            max_exectime = self.get_max_exectime (workitem_pipelinestages) * 2

            if max_exectime == 0:#first time execution
                max_exectime = rmanager.get_max_exectime (workitem_pipelinestages, self.id, active=True) * 2

            if max_exectime == 0:#no one has completed their execution
                max_exectime = 20 * 60

            return max_exectime

        if resourcetype == 'GPU' and self.gpu != None and self.gpu.workqueue.is_empty () == False:
            workitem = self.gpu.workqueue.get_workitem ()
            workitem_pipelinestages = workitem.get_pipelinestage ()

            max_exectime = self.get_max_exectime (workitem_pipelinestages) * 2

            if max_exectime == 0:#first time execution
                max_exectime = rmanager.get_max_exectime (workitem_pipelinestages, self.id, active=True) * 2

            if max_exectime == 0:#no one has completed their execution
                max_exectime = 15 * 60

            return max_exectime

    def get_status (self, rmanager, threads, outputfile):
        #print ('get_status ():', self.id)
        #first cpu
        if self.cpu != None and self.cpu.workqueue.is_empty () == False:
            workitem = self.cpu.workqueue.get_workitem ()
            ret, start_time, end_time, status, r_timetaken = workitem.probe_status (threads[0], outputfile)
            if ret == True:
                if status == 'SUCCESS':
                    #print ('cpu workitem complete')
                    self.cpu.set_busy (False)
                    self.cpu.set_last_completion_time (end_time)
                    self.cpu.set_idle_start_time (end_time)
                    self.add_count (workitem.get_pipelinestage ())
                    rmanager.add_exectime (self.cpu.name, workitem.get_pipelinestage (), start_time, end_time)
                    self.add_exectime(workitem.get_pipelinestage (), start_time,
                                      end_time)
                    return end_time - start_time
                elif status == 'FAILED':
                    #print ('cpu workitem failed')
                    self.cpu.set_busy (False)
                    self.cpu.set_last_completion_time (end_time)
                    return None
                elif status == 'CANCELLED':
                    #print ('cpu workitem cancelled')
                    self.cpu.set_busy (False)
                    self.cpu.set_last_completion_time (end_time)
                    return None

                #self.add_transfer_time (pmanager.encode_pipeline_stages(workitem.get_pipelinestages ()), r_timetaken)

        #now gpu
        if self.gpu != None and self.gpu.workqueue.is_empty () == False:
            workitem = self.gpu.workqueue.get_workitem ()
            ret, start_time, end_time, status, r_timetaken = workitem.probe_status (threads[1], outputfile)
            if ret == True:
                if status == 'SUCCESS':
                    #print ('gpu workitem complete')
                    self.gpu.set_busy (False)
                    self.gpu.set_last_completion_time (end_time)
                    self.gpu.set_idle_start_time(end_time)
                    self.add_count (workitem.get_pipelinestage ())
                    rmanager.add_exectime (self.gpu.name, workitem.get_pipelinestage (), start_time, end_time)
                    self.add_exectime(workitem.get_pipelinestage (), start_time,
                                      end_time)
                    return end_time - start_time
                elif status == 'FAILED':
                    #print ('gpu workitem failed')
                    self.gpu.set_busy (False)
                    self.gpu.set_last_completion_time (end_time)
                    return None

                elif status == 'CANCELLED':
                    #print ('gpu workitem cancelled')
                    self.gpu.set_busy (False)
                    self.gpu.set_last_completion_time (end_time)
                    return None

        return None
                #self.add_transfer_time (pmanager.encode_pipeline_stages(workitem.get_pipelinestages ()), r_timetaken)

    def get_last_completion_time (self, resourcetype):
        if resourcetype == 'CPU' and self.cpu != None:
            return self.cpu.get_last_completion_time ()

        if resourcetype == 'GPU' and self.gpu != None:
            return self.gpu.get_last_completion_time ()

    def add_workitem (self, workitem, resourcetype):
        #print ('add_workitem ():', self.id, workitem.id, workitem.version)
        if resourcetype == 'CPU':
            if self.cpu == None:
                #print (self.id, 'CPU not available')
                return
            self.cpu.workqueue.add_workitem (workitem)
            self.executionqueues[0] = workitem.version

        if resourcetype == 'GPU':
            if self.gpu == None:
                #print (self.id, 'GPU not available')
                return
            self.gpu.workqueue.add_workitem (workitem)
            self.executionqueues[1] = workitem.version

    def pop_if_complete (self, resourcetype):
        if resourcetype == 'CPU' and self.cpu == None:
            #print (self.id, 'CPU not available')
            return None

        if resourcetype == 'GPU' and self.gpu == None:
            #print (self.id, 'GPU not available')
            return None

        if resourcetype == 'CPU' and self.cpu.workqueue.is_empty () == False:
            if self.cpu.workqueue.get_workitem ().is_complete () == True:
                #print (self.id, 'CPU workitem complete')
                workitem = self.cpu.workqueue.pop_workitem ()
                self.executionqueues[0] = -1
                return workitem

        if resourcetype == 'GPU' and self.gpu.workqueue.is_empty () == False:
            if self.gpu.workqueue.get_workitem ().is_complete () == True:
                #print (self.id, 'GPU workitem complete')
                workitem = self.gpu.workqueue.pop_workitem ()
                self.executionqueues[1] = -1
                return workitem

        return None

    def get_workitem (self, resourcetype):
        if resourcetype == 'CPU' and self.cpu == None:
            return None

        if resourcetype == 'GPU' and self.gpu == None:
            return None

        if resourcetype == 'CPU' and self.cpu.workqueue.is_empty () == False:
            workitem = self.cpu.workqueue.get_workitem()
            return workitem

        if resourcetype == 'GPU' and self.gpu.workqueue.is_empty () == False:
            workitem = self.gpu.workqueue.get_workitem()
            return workitem

        return None

    def get_hostname (self):
        return self.hostname

    def get_count (self, pipelinestages):
        if pipelinestages not in self.counts:
            return 0
        return self.counts[pipelinestages]

    def add_count (self, pipelinestages):
        if pipelinestages in self.counts:
            self.counts[pipelinestages] += 1
        else:
            self.counts[pipelinestages] = 1

    def add_exectime (self, pipelinestages, starttime, endtime):
        seconds = endtime - starttime

        if pipelinestages not in self.exectimes:
            self.exectimes[pipelinestages] = [seconds, 1]
            self.max_exectimes[pipelinestages] = seconds
        else:
            if seconds > self.max_exectimes[pipelinestages]:
                self.max_exectimes[pipelinestages] = seconds
            avg_time = self.exectimes[pipelinestages][0]
            count = self.exectimes[pipelinestages][1]
            new_avg_time = ((avg_time * count) + seconds) / (count + 1)
            self.exectimes[pipelinestages] = [new_avg_time, count + 1]

    def add_transfer_time (self, pipelinestages, timetaken):
        if timetaken == 0:
            return

        if pipelinestages not in self.transfertimes:
            self.transfertimes[pipelinestages] = [timetaken, 1]
        else:
            avg_time = self.transfertimes[pipelinestages][0]
            count = self.transfertimes[pipelinestages][1]
            new_transfer_time = ((avg_time * count) + timetaken) / (count + 1)
            self.transfertimes[pipelinestages] = [new_transfer_time, count + 1]

    def get_max_exectime (self, pipelinestages):
        if pipelinestages not in self.max_exectimes:
            return 0
        else:
            return self.max_exectimes[pipelinestages]

    def get_exectime (self, pipelinestages, resourcetype):
        if pipelinestages not in self.exectimes:
            if resourcetype == 'CPU':
                resource_name = self.cpu.name
            else:
                resource_name = self.gpu.name
            exectime = self.rmanager.get_exectime (resource_name, pipelinestages)
            return exectime
        else:
            return self.exectimes[pipelinestages][0]

    def get_throughput (self, pipelinestages, resourcetype):
        exectime = self.get_exectime(pipelinestages, resourcetype)

        return 1 / exectime

    def get_exectime_pipelinestages (self, pmanager, pipelinestages, resourcetype):
        if resourcetype == 'CPU' and self.cpu != None:
            encoded_pipelinestages = pmanager.encode_pipeline_stages (pipelinestages)
            exectime = self.get_exectime(encoded_pipelinestages, resourcetype)
            if exectime == 0:
                exectime = self.rmanager.get_exectime(self.cpu.name, pipelinestages)
                if exectime == 0:
                    return None
            else:
                return exectime
        else:
            return None

        if resourcetype == 'GPU' and self.gpu != None:
            encoded_pipelinestages = pmanager.encode_pipeline_stages (pipelinestages)
            exectime = self.get_exectime(encoded_pipelinestages, resourcetype)
            if exectime == 0:
                exectime = self.rmanager.get_exectime (self.gpu.name, pipelinestages)
                if exectime == 0:
                    return None
            else:
                return exectime
        else:
            return None

    def get_exectime_current (self, pmanager, resourcetype):

        if resourcetype == 'CPU' and self.cpu != None:
            if self.cpu.workqueue.is_empty () == False:
                workitem = self.cpu.workqueue.get_workitem ()
                workitem_pipelinestages = workitem.get_pipelinestages ()
                encoded_workitem_pipelinestages = pmanager.encode_pipeline_stages (workitem_pipelinestages)
                exectime = self.get_exectime(encoded_workitem_pipelinestages, resourcetype)
                if exectime == 0:
                    exectime = self.rmanager.get_exectime(self.cpu.name, encoded_workitem_pipelinestages)
                    if exectime == 0:
                        return None
                else:
                    return exectime
            else:
                return None

        if resourcetype == 'GPU' and self.gpu != None:
            if self.gpu.workqueue.is_empty () == False:
                workitem = self.gpu.workqueue.get_workitem ()
                workitem_pipelinestages = workitem.get_pipelinestages ()
                encoded_workitem_pipelinestages = pmanager.encode_pipeline_stages (workitem_pipelinestages)
                exectime = self.get_exectime(encoded_workitem_pipelinestages, resourcetype)
                if exectime == 0:
                    exectime = self.rmanager.get_exectime(self.gpu.name, encoded_workitem_pipelinestages)
                    if exectime == 0:
                        return None
                else:
                    return exectime
            else:
                return None

    def get_transfertime (self, pipelinestages):
        if pipelinestages not in self.exectimes:
            return 0
        else:
            return self.transfertimes[pipelinestages][0]

    def get_work_left (self, resourcetype, current_time):
        if resourcetype == 'CPU' and self.cpu != None:
            if self.cpu.workqueue.is_empty () == False:
                workitem = self.cpu.workqueue.get_workitem ()
                workitem_pipelinestages = workitem.get_pipelinestage ()

                exectime = self.get_exectime (workitem_pipelinestages, resourcetype)

                if exectime == 0:
                    exectime = self.rmanager.get_exectime(self.cpu.name, workitem_pipelinestages)
                    if exectime == 0:
                        return None
                else:
                    work_remaining = (exectime - (current_time - workitem.scheduletime))/exectime
                    if work_remaining < 0:
                        work_remaining = 0
                return work_remaining

        if resourcetype == 'GPU' and self.gpu != None:
            if self.gpu.workqueue.is_empty () == False:
                workitem = self.gpu.workqueue.get_workitem ()
                workitem_pipelinestages = workitem.get_pipelinestage ()

                exectime = self.get_exectime (workitem_pipelinestages, resourcetype)

                if exectime == 0:
                    exectime = self.rmanager.get_exectime(self.gpu.name, workitem_pipelinestages)
                    if exectime == 0:
                        return None
                else:
                    work_remaining = (exectime - (current_time - workitem.scheduletime))/exectime
                    if work_remaining < 0:
                        work_remaining = 0

                return work_remaining

        return None

    def get_time_left (self, resourcetype, current_time):
        if resourcetype == 'CPU' and self.cpu != None:
            if self.cpu.workqueue.is_empty () == False:
                workitem = self.cpu.workqueue.get_workitem ()
                workitem_pipelinestages = workitem.get_pipelinestages ()

                exectime = self.get_exectime (workitem_pipelinestages, resourcetype)

                print('get_time_left cpu', exectime, current_time, workitem.scheduletime)

                if exectime == 0:
                    exectime = self.rmanager.get_exectime(self.cpu.name, workitem_pipelinestages)
                    if exectime == 0:
                        return None
                else:
                    work_remaining = (exectime - (current_time - workitem.scheduletime))
                    if work_remaining < 0:
                        work_remaining = 0
                return work_remaining

        if resourcetype == 'GPU' and self.gpu != None:
            if self.gpu.workqueue.is_empty () == False:
                workitem = self.gpu.workqueue.get_workitem ()
                workitem_pipelinestages = workitem.get_pipelinestages ()

                exectime = self.get_exectime (workitem_pipelinestages, resourcetype)

                print('get_time_left gpu', exectime, current_time, workitem.scheduletime)

                if exectime == 0:
                    exectime = self.rmanager.get_exectime(self.gpu.name, workitem_pipelinestages)
                    if exectime == 0:
                        return None
                else:
                    work_remaining = (exectime - (current_time - workitem.scheduletime))
                    if work_remaining < 0:
                        work_remaining = 0

                return work_remaining

        return None

class ResourceManager:
    def __init__ (self, env):
        self.resourcetypeinfo = {}
        self.exectimes = {}
        self.max_exectimes = {}
        self.active_cpunodes_count = 0
        self.active_gpunodes_count = 0
        self.backup_cpunodes_count = 0
        self.backup_gpunodes_count = 0
        self.cpuid_counter = 0
        self.gpuid_counter = 0
        self.env = env
        self.total_cpu_cost = 0
        self.total_gpu_cost = 0
        self.active_pool_nodes = []
        self.backup_pool_nodes = []
        self.exploration_resources = []
        self.pinned_resources = []
        self.domain_resource_map = {}

    def get_startup_time (self, resourcetype, provision_type):
        return float (self.resourcetypeinfo[provision_type][resourcetype]['startuptime'] / 3600)

    def get_availability (self, resourcetype, provision_type):
        return self.resourcetypeinfo[provision_type][resourcetype]['availability']

    def request_resource (self, resourcename):
        if rand.random() > 0.5:
            return True
        return False

    def add_resource (self, domain, cpuok, gpuok, resource_type, provision_type, active_pool, bidding_price, pipelinestageindex):

        print ('add resource ()', domain.name, cpuok, gpuok, resource_type, provision_type, active_pool, bidding_price, pipelinestageindex)

        if provision_type == 'on_demand':
            on_demand = True
        else:
            on_demand = False
        cost, provision_time = domain.provision_resource(resource_type, on_demand, bidding_price)

        if provision_time == None or cost == None:
            print('resource provision failed', resource_type, provision_type, cost)
            return None, None

        if provision_type not in self.resourcetypeinfo:
            self.resourcetypeinfo[provision_type] = {}

        if resource_type not in self.resourcetypeinfo[provision_type]:
            self.resourcetypeinfo[provision_type][resource_type] = {}

        if cpuok == True:
            resource = Resource ('c' + str(self.cpuid_counter), self, provision_type, self.env, pipelinestageindex, domain.id)
            resource.add_cpu (resource_type, cost, self.env.now)
            resource.pfs = domain.get_pfs_handle()

            if active_pool == True:
                self.active_pool_nodes.append(resource)
                self.active_cpunodes_count += 1
            else:
                self.backup_pool_nodes.append(resource)
                self.backup_cpunodes_count += 1
            self.cpuid_counter += 1

            self.resourcetypeinfo[provision_type][resource_type]['resourcetype'] = 'CPU'
        elif gpuok == True:
            resource = Resource('g' + str(self.gpuid_counter), self, provision_type, self.env, pipelinestageindex, domain.id)
            resource.add_gpu(resource_type, cost, self.env.now)
            resource.pfs = domain.get_pfs_handle()

            if active_pool == True:
                self.active_pool_nodes.append(resource)
                self.active_gpunodes_count += 1
            else:
                self.backup_pool_nodes.append(resource)
                self.backup_gpunodes_count += 1
            self.gpuid_counter += 1

        self.resourcetypeinfo[provision_type][resource_type]['provisiontime'] = provision_time
        self.resourcetypeinfo[provision_type][resource_type]['availability'] = 1.0
        self.resourcetypeinfo[provision_type][resource_type]['cost'] = cost

        if 'count' not in self.resourcetypeinfo[provision_type][resource_type]:
            self.resourcetypeinfo[provision_type][resource_type]['count'] = {}
            self.resourcetypeinfo[provision_type][resource_type]['count']['time'] = [self.env.now]
            self.resourcetypeinfo[provision_type][resource_type]['count']['count'] = [1]
        else:
            self.resourcetypeinfo[provision_type][resource_type]['count']['time'].append(self.env.now)
            self.resourcetypeinfo[provision_type][resource_type]['count']['count'].append(
                self.resourcetypeinfo[provision_type][resource_type]['count']['count'][-1] + 1)

        self.pinned_resources.append(resource)

        if str(domain.id) not in self.domain_resource_map:
            self.domain_resource_map[str(domain.id)] = []

        self.domain_resource_map[str(domain.id)].append (resource.id)

        return resource, provision_time

    def delete_resource (self, resourcetype, resource_id, active):
        if active == True:
            nodes = self.active_pool_nodes
        else:
            nodes = self.backup_pool_nodes
        print('delete resource ()', resource_id, resourcetype, active, len(nodes))
        node_index = 0
        for node in nodes:
            if node.id == resource_id:
                if node.active == True:
                    provision_type = node.provision_type
                    if resourcetype == 'CPU':
                        self.resourcetypeinfo[provision_type][node.cpu.name]['count']['time'].append(self.env.now)
                        self.resourcetypeinfo[provision_type][node.cpu.name]['count']['count'].append (self.resourcetypeinfo[provision_type][node.cpu.name]['count']['count'][-1] - 1)
                        self.total_cpu_cost += (self.env.now - node.acquisition_time) * node.cpu.cost
                    else:
                        self.resourcetypeinfo[provision_type][node.gpu.name]['count']['time'].append (self.env.now)
                        self.resourcetypeinfo[provision_type][node.gpu.name]['count']['count'].append (self.resourcetypeinfo[provision_type][node.gpu.name]['count']['count'][-1] - 1)
                        self.total_gpu_cost += (self.env.now - node.acquisition_time) * node.gpu.cost
                break
            node_index += 1

        if node_index < len (nodes):

            node = nodes.pop (node_index)

            self.pinned_resources.remove(node)

            print('delete_resource () pinned', len(self.pinned_resources))

            print ('delete_resource () domain resource map', self.domain_resource_map)

            self.domain_resource_map[str(node.domain_id)].remove (node.id)

            if resourcetype == 'CPU':
                if active == True:
                    self.active_cpunodes_count -= 1
                else:
                    self.backup_cpunodes_count -= 1
            else:
                if active == True:
                    self.active_gpunodes_count -= 1
                else:
                    self.backup_gpunodes_count -= 1

    def add_pinned_resource (self, node_id):
        node = self.get_resource (node_id, True)
        self.pinned_resources.append (node)

    def get_total_cost (self):
        return self.total_cpu_cost, self.total_gpu_cost

    def get_resourcetype_info_all (self):
        return self.resourcetypeinfo

    def get_resourcetype_info (self, resourcetype, infotype, provision_type):
        if provision_type not in self.resourcetypeinfo:
            return None
        return self.resourcetypeinfo[provision_type][resourcetype][infotype]

    def get_max_exectime (self, pipelinestages, resource_id, active):
        #TODO: this function is completely wrong, use resourcename and resourcetype structure to get max exectime
        resources = self.get_resources ('active', active)

        max_exectime = 0
        for resource in resources:
            if resource.id == resource_id:
                continue

            if resource.get_max_exectime (pipelinestages) > max_exectime:
                max_exectime = resource.get_max_exectime (pipelinestages)

        return max_exectime

    def get_cpu_resources_count (self, active):
        if active == True:
            return self.active_cpunodes_count
        else:
            return self.backup_cpunodes_count

    def get_gpu_resources_count (self, active):
        if active == True:
            return self.active_gpunodes_count
        else:
            return self.backup_gpunodes_count

    def add_exectime (self, resourcename, pipelinestages, starttime, endtime):
        seconds = endtime - starttime

        if resourcename not in self.exectimes:
            self.exectimes[resourcename] = {}
            self.max_exectimes[resourcename] = {}
            if pipelinestages not in self.exectimes:
                self.exectimes[resourcename][pipelinestages] = [seconds, 1]
                self.max_exectimes[resourcename][pipelinestages] = seconds
        else:
            if pipelinestages not in self.exectimes:
                self.exectimes[resourcename][pipelinestages] = [seconds, 1]
                self.max_exectimes[resourcename][pipelinestages] = seconds
            else:
                if seconds > self.max_exectimes[resourcename][pipelinestages]:
                    self.max_exectimes[resourcename][pipelinestages] = seconds
                avg_time = self.exectimes[resourcename][pipelinestages][0]
                count = self.exectimes[resourcename][pipelinestages][1]
                new_avg_time = ((avg_time * count) + seconds) / (count + 1)
                self.exectimes[resourcename][pipelinestages] = [new_avg_time, count + 1]

        #print (self, self.exectimes)

    def get_exectime (self, resourcename, pipelinestages):
        if resourcename not in self.exectimes:
            return 0
        if pipelinestages not in self.exectimes[resourcename]:
            return 0

        return self.exectimes[resourcename][pipelinestages][0]

    def get_pipelinestage_weights (self, resourcetype, no_of_pipelinestages):
        total_execution_times = {}
        count = 0
        for resource_name in self.exectimes.keys ():
            resourcename_execution_times = []
            if self.get_resourcetype_info(resource_name, 'resourcetype', 'on_demand') == resourcetype:
                if len (list (self.exectimes[resource_name].keys ())) < no_of_pipelinestages:
                    continue
                for pipelinestage in self.exectimes[resource_name].keys ():
                    resourcename_execution_times.append(self.exectimes[resource_name][pipelinestage][0] * self.exectimes[resource_name][pipelinestage][1])

                total_execution_times[resource_name] = resourcename_execution_times

        weights = [0 for i in range (0, no_of_pipelinestages)]

        for key in total_execution_times.keys():
            index = 0
            for exectimes in total_execution_times[key]:
                weights[index] += exectimes
                index +=1

        total_no_of_resource_types = len(list(total_execution_times.keys()))

        weights = [weight/total_no_of_resource_types for weight in weights]

        total_weights = sum(weights)

        weights = [weight/total_weights for weight in weights]

        print (resourcetype, weights)

        return weights

    def get_resource (self, resource_id, active):
        if active == True:
            nodes = self.active_pool_nodes
        else:
            nodes = self.backup_pool_nodes
        for node in nodes:
            if node.id == resource_id:
                return node
        return None

    def get_resources (self, pool, active):
        ret = []
        if pool == 'active':
            nodes = self.active_pool_nodes
        else:
            nodes = self.backup_pool_nodes
        for resource in nodes:
            if resource.active == active:
                ret.append(resource)
        return ret

    def get_resources_type (self, resourcetype, active):
        resources = []
        if active == True:
            nodes = self.active_pool_nodes
        else:
            nodes = self.backup_pool_nodes
        for resource in nodes:
            if resourcetype == 'CPU' and resource.cpu != None:
                resources.append(resource)
            elif resourcetype == 'GPU' and resource.gpu != None:
                resources.append(resource)
        return resources

    def get_resource_names (self, resourcetype):
        provision_type = 'on_demand'
        resource_names = []
        for resource_name in self.resourcetypeinfo[provision_type].keys ():
            if self.resourcetypeinfo[provision_type][resource_name]['resourcetype'] == resourcetype:
                resource_names.append(resource_name)
        return resource_names