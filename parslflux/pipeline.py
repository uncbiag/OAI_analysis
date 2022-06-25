import yaml
import sys
import statistics
import copy
import math
from plots.plot_prediction_sim import plot_prediction_sim_0
from parslfluxsim.resources_sim import Resource
from parslfluxsim.bagofworkitems_sim import BagOfWorkItems
from parslfluxsim.workitem_sim import WorkItem

class PipelineStage:
    def __init__ (self, stageindex, name, resourcetype, priority, output_size, rmanager, batchsize):
        index = 0
        self.name = name
        self.index = stageindex
        self.resourcetype = resourcetype
        self.priority = priority
        self.output_size = output_size
        self.pinned_resources = []
        self.rmanager = rmanager
        self.exec_parents = []
        self.exec_children = []
        self.data_parents = []
        self.data_children = []
        self.bagofworkitems = BagOfWorkItems (self.index, self.resourcetype)
        self.batchsize = batchsize
        self.total_complete = 0

    def add_completion (self, count):
        self.total_complete += count

    def get_pending_workitems (self):
        return self.batchsize - self.total_complete

    def populate_bagofworkitems (self, imanager):

        if len (self.exec_parents) > 0:
            return
        count = 0
        while imanager.get_remaining_count () > 0:
            images = imanager.get_images (1)
            image_key = list (images.keys())[0]

            new_workitem = WorkItem (image_key, images[image_key], None, self, None, self.resourcetype, \
                                      self.index, '')

            self.bagofworkitems.add_workitem(new_workitem)
            count += 1

        imanager.add_back_images (count)

    def add_pinned_resource (self, resource_id):
        self.pinned_resources.append (resource_id)

    def remove_pinned_resource (self, resource_id):
        self.pinned_resources.remove(resource_id)

    def get_pinned_resources (self, rmanager, status):
        results = []
        for resource_id in self.pinned_resources:
            resource = rmanager.get_resource (resource_id, True)
            if resource.active == status:
                results.append (resource_id)

        return results

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

    def get_parents (self, type):
        if type == 'exec':
            return self.exec_parents
        else:
            return self.data_parents

    def get_current_throughput (self, phase_index):
        #print ('get_current_throughput ()', self.name)
        current_executors = self.pinned_resources

        thoughput_list = []
        for resource_id in current_executors:
            #print('get_current_throughput ()', resource_id)
            resource = self.rmanager.get_resource (resource_id, active=True)
            if resource == None:
                continue
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

    def print_data (self):
        print (self.name, self.index, self.resourcetype, self.data_parents, self.exec_parents, self.data_children, self.exec_children)

class PipelineManager:
    def __init__ (self, pipelinefile, batchsize, max_images):
        self.pipelinefile = pipelinefile
        self.pipelinestages = []
        self.pipelinestages_dict = {}
        self.batchsize = batchsize
        self.last_last_phase_closed_index = -1
        self.last_first_phase_closed_index = -1
        self.no_of_columns = int(math.ceil(max_images / batchsize))
        self.prediction_times = []
        self.prediction_idle_periods = {}
        self.max_images = max_images
        self.effective_throughput_record = {}
        self.data_throughput_record = {}
        self.throughput_record = {}

    def get_throughput_record (self):
        return self.throughput_record, self.effective_throughput_record

    def get_data_throughput_record (self):
        print (self.data_throughput_record)
        return self.data_throughput_record

    def performance_to_cost_ranking_pipelinestage (self, rmanager, pipelinestageindex):

        weighted_performance_to_cost_ratio = {}

        pipelinestage = self.pipelinestages[pipelinestageindex]

        resourcetnames = rmanager.get_resource_names(pipelinestage.resourcetype)

        for resource_name in resourcetnames:
            performance_to_cost_ratio = 0
            exectime = rmanager.get_exectime(resource_name, pipelinestage.name)
            throughput = 1 / exectime
            ret = rmanager.get_resourcetype_info(resource_name, 'cost', 'on_demand')
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

        weighted_performance_to_cost_ratio_ranking = dict(
            sorted(weighted_performance_to_cost_ratio.items(), key=lambda item: item[1], reverse=True))

        return weighted_performance_to_cost_ratio_ranking

    def calculate_idleness_cost (self, rmanager, env):
        idleness_cost_dict = {}

        for pipelinestage in self.pipelinestages:

            idleness_cost_dict[str(pipelinestage.index)] = {}

            available_throughput = self.throughput_record[str(pipelinestage.index)][-1][1]

            effective_throughput = self.effective_throughput_record[str(pipelinestage.index)][-1][1]

            total_throughput_waste = available_throughput - effective_throughput

            pinned_resources = pipelinestage.get_pinned_resources (rmanager, True)

            for pinned_resource_id in pinned_resources:
                pinned_resource = rmanager.get_resource (pinned_resource_id, active=True)

                pinned_resource_throughput = pinned_resource.get_throughput (pipelinestage.name, pipelinestage.resourcetype)

                pinned_resource_throughput_waste = float (pinned_resource_throughput / available_throughput) * total_throughput_waste

                pinned_resource_cost = pinned_resource.get_cost (pipelinestage.resourcetype)

                pinned_resource_wastage_cost = (pinned_resource_throughput_waste / pinned_resource_throughput) * pinned_resource_cost

                idleness_cost_dict[str(pipelinestage.index)][pinned_resource_id] = pinned_resource_wastage_cost

                print ('resource_wastage_cost ()', pipelinestage.index, pinned_resource_id, pinned_resource_throughput, pinned_resource_throughput_waste,
                       pinned_resource_cost, pinned_resource_wastage_cost)

        for pipelinestageindex in idleness_cost_dict.keys ():
            for resource_id in idleness_cost_dict[pipelinestageindex].keys ():
                print ('resource_wastage_cost ()', pipelinestageindex, resource_id, idleness_cost_dict[pipelinestageindex][resource_id])

    def calculate_pipeline_stats (self, env):
        effective_throughput_dict = {}
        throughput_dict = {}

        root_pipelinestages = []

        for pipelinestage in self.pipelinestages:
            pipelinestage_parents = pipelinestage.get_parents ('data')

            if len (pipelinestage_parents) == 0:
                root_pipelinestages.append(pipelinestage)
                #print ('calculate_pipeline_stats () root stage', pipelinestage.name)

        parent_effective_throughputs = {}
        for root_pipelinestage in root_pipelinestages:

            to_be_traversed = []

            to_be_traversed.append (root_pipelinestage)

            while len(to_be_traversed) > 0:
                current_pipelinestage = to_be_traversed.pop(0)

                current_throughput = current_pipelinestage.get_current_throughput (0)
                if str(current_pipelinestage.index) not in throughput_dict:
                    throughput_dict[str(current_pipelinestage.index)] = current_throughput



                if str(current_pipelinestage.index) not in parent_effective_throughputs:
                    effective_throughput_dict[str(current_pipelinestage.index)] = current_throughput
                    #print('calculate_pipeline_stats () 1', current_pipelinestage.name, current_throughput, effective_throughput_dict[str(current_pipelinestage.index)])
                else:
                    if str(current_pipelinestage.index) in effective_throughput_dict:
                        if effective_throughput_dict[str(current_pipelinestage.index)] > parent_effective_throughputs[str(current_pipelinestage.index)]:
                            effective_throughput_dict[str(current_pipelinestage.index)] = parent_effective_throughputs[str(current_pipelinestage.index)]
                            #print('calculate_pipeline_stats () 2', current_pipelinestage.name, current_throughput,
                            #      effective_throughput_dict[str(current_pipelinestage.index)])
                        else:
                            if effective_throughput_dict[str(current_pipelinestage.index)] > current_throughput:
                                effective_throughput_dict[str(current_pipelinestage.index)] = current_throughput
                                #print('calculate_pipeline_stats () 3', current_pipelinestage.name, current_throughput,
                                #      effective_throughput_dict[str(current_pipelinestage.index)])
                    else:
                        if current_throughput > parent_effective_throughputs[str(current_pipelinestage.index)]:
                            effective_throughput_dict[str(current_pipelinestage.index)] = parent_effective_throughputs[str(current_pipelinestage.index)]
                            #print('calculate_pipeline_stats () 4', current_pipelinestage.name, current_throughput,
                            #      effective_throughput_dict[str(current_pipelinestage.index)])
                        else:
                            effective_throughput_dict[str(current_pipelinestage.index)] = current_throughput
                            #print('calculate_pipeline_stats () 5', current_pipelinestage.name, current_throughput,
                            #      effective_throughput_dict[str(current_pipelinestage.index)])

                children_pipelinestages = current_pipelinestage.get_children('data')

                for children_pipelinestage in children_pipelinestages:
                    #print ('current', current_pipelinestage.name, 'child', children_pipelinestage.name)
                    if str(children_pipelinestage.index) in parent_effective_throughputs:
                        if parent_effective_throughputs[str(children_pipelinestage.index)] > effective_throughput_dict[str(current_pipelinestage.index)]:
                            parent_effective_throughputs[str(children_pipelinestage.index)] = effective_throughput_dict[str(current_pipelinestage.index)]

                    else:
                        parent_effective_throughputs[str(children_pipelinestage.index)] = effective_throughput_dict[str(current_pipelinestage.index)]

                    to_be_traversed.append(children_pipelinestage)

        pipelinestageindex = 0

        while pipelinestageindex < len (self.pipelinestages):
            current_pipelinestage = self.pipelinestages[pipelinestageindex]

            if str(current_pipelinestage.index) not in self.effective_throughput_record.keys():
                self.effective_throughput_record[str(current_pipelinestage.index)] = []

            self.effective_throughput_record[str(current_pipelinestage.index)].append ([env.now, effective_throughput_dict[str(current_pipelinestage.index)]])

            if str(current_pipelinestage.index) not in self.throughput_record.keys():
                self.throughput_record[str(current_pipelinestage.index)] = []

            self.throughput_record[str(current_pipelinestage.index)].append ([env.now, throughput_dict[str(current_pipelinestage.index)]])

            print ('calculate_pipeline_stats ()', current_pipelinestage.name,
                   effective_throughput_dict[str(pipelinestageindex)], throughput_dict[str(pipelinestageindex)])

            pipelinestage_throughput = effective_throughput_dict[str(current_pipelinestage.index)]
            children_pipelinestages = current_pipelinestage.get_children('data')

            max_data_throughput = 0

            for children_pipelinestage in children_pipelinestages:
                child_throughput = effective_throughput_dict[str(children_pipelinestage.index)]

                if child_throughput < pipelinestage_throughput:
                    data_throughput = float (((pipelinestage_throughput - child_throughput) * pipelinestage.output_size) / 1024)

                    if data_throughput > max_data_throughput:
                        max_data_throughput = data_throughput

            if str(current_pipelinestage.index) not in self.data_throughput_record.keys():
                self.data_throughput_record[str(current_pipelinestage.index)] = []

            self.data_throughput_record[str(current_pipelinestage.index)].append ([env.now, max_data_throughput])

            pipelinestageindex += 1

        return

    def reconfiguration (self, rmanager, env):
        self.calculate_pipeline_stats (env)
        self.calculate_idleness_cost (rmanager, env)

    def parse_pipelines (self, rmanager):
        pipelinedatafile = open(self.pipelinefile)
        pipelinedata = yaml.load(pipelinedatafile, Loader=yaml.FullLoader)

        index = 0
        for pipelinestage_node in pipelinedata['pipelinestages']:
            pipelinestage_name = pipelinestage_node['name']
            pipelinestage_resourcetype = pipelinestage_node['resourcetype']
            pipelinestage_priority = int (pipelinestage_node['priority'])
            pipelinestage_output_size = pipelinestage_node['output_size']

            new_pipelinestage = PipelineStage(index, pipelinestage_name, pipelinestage_resourcetype, pipelinestage_priority, pipelinestage_output_size, rmanager, self.batchsize)

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

        #for pipelinestage in self.pipelinestages:
        #    pipelinestage.print_data ()

    def print_stage_queue_data_2 (self, rmanager, pfs):
        plot_data = {}
        for pipelinestage in self.pipelinestages:
            plot_data[pipelinestage.name] = []
            for phase in pipelinestage.phases:
                plot_data[pipelinestage.name].append ([phase.queue_snapshots, phase.starttime, phase.endtime, phase.predictions])

        plot_prediction_sim_0 (self, rmanager, plot_data, pfs)


    def get_all_pipelinestages (self):
        return self.pipelinestages

    def get_first_pipelinestage (self):
        return self.pipelinestages[0]

    def get_next_pipelinestages (self, current_pipelinestage):
        children = current_pipelinestage.get_children ('exec')

        return children

    def get_pipelinestage (self, index):
        return self.pipelinestages[index]


if __name__ == "__main__":
    pipelinefile = sys.argv[1]
    p = PipelineManager(pipelinefile)
    p.parse_pipelines ()
    p.print_data ()
