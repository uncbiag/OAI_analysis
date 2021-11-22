import yaml
import sys
from parslfluxsim.resources_sim import Resource

class Phase:
    def __init__(self, pipelinestage, starttime):
        self.pipelinestage = pipelinestage
        self.starttime = starttime
        self.count = 0
        self.total_count = 0
        self.timestamps = {}
        self.endtime = -1
        self.active = True
        self.first_stage_targets = []
        self.target = -1
        self.current_executors = []
        self.workitems = []

    def set_target (self, target):
        self.target = target

    def add_executor (self, resource):
        if resource.id not in self.current_executors:
            self.current_executors.append(resource.id)

    def remove_executor (self, resource):
        if resource.id in self.current_executors:
            self.current_executors.remove(resource.id)

    def add_workitem (self, workitem, currenttime):
        self.count += 1
        self.total_count += 1
        self.timestamps[workitem.id + ':' + str (currenttime)] = self.count
        self.workitems.append(workitem.id)
        print(self.pipelinestage, 'add workitem', workitem.id, self.count)

    def remove_workitem (self, currenttime, workitem):
        self.count -= 1
        self.timestamps[workitem.id + ':' + str (currenttime)] = self.count
        self.workitems.remove (workitem.id)
        print(self.pipelinestage, 'remove workitem', workitem.id, self.count)

    def close_phase (self, nowtime):
        self.active = False
        self.endtime = nowtime
        print('phase closed', self.pipelinestage, self.starttime, self.endtime, self.total_count)

    def print_data (self):
        print (self.pipelinestage, self.starttime, self.endtime, self.total_count)
        print (self.timestamps)


class PipelineStage:
    def __init__ (self, index, pipelinestage):
        self.name = pipelinestage['name']
        self.index = index
        self.resourcetype = pipelinestage['resource']
        self.phases = []

    def add_phase (self, currenttime):
        print(self.name, 'add phase', currenttime)
        new_phase = Phase(self.name, currenttime)
        self.phases.append(new_phase)
        return new_phase

    def get_resourcetype (self):
        return self.resourcetype

    def get_index (self):
        return self.index

    def get_name (self):
        return self.name

    def get_phase (self, workitem):
        index = 0
        for phase in self.phases:
            if phase.active == False:
                index += 1
                continue
            if workitem.id in phase.workitems:
                return phase, index
            index += 1
        return None, None

    def get_phase_index (self, index):
        print('get_phase_index', len(self.phases), index)
        if index <= (len(self.phases) - 1):
            return self.phases[index]

        return None

    def get_latest_phase (self):
        if len (self.phases) < 1:
            return None

        return self.phases[-1]

    def add_workitem_index (self, index, workitem, current_time):
        if index <= (len (self.phases) - 1):
            if self.phases[index].active == False:
                phase = self.add_phase(current_time)
            else:
                phase = self.phases[index]

            phase.add_workitem (workitem, current_time)
        else:
            phase = self.add_phase(current_time)
            phase.add_workitem(workitem, current_time)

        print(self.name, 'add workitem index', workitem.id, len(self.phases) - 1, phase.count)

    def add_workitem (self, workitem, current_time):
        if len (self.phases) > 0:
            latest_phase = self.phases[-1]
            if latest_phase.active == False:
                latest_phase = self.add_phase(current_time)
        else:
            latest_phase = self.add_phase(current_time)

        latest_phase.add_workitem(workitem, current_time)
        print (self.name, 'add workitem', workitem.id, len (self.phases) - 1, latest_phase.count)

    def close_phase (self, currenttime):
        current_phase = self.phases[-1]
        current_phase.set_endtime (currenttime)

    def add_executor (self, workitem, resource):
        current_phase, index = self.get_phase(workitem)
        if current_phase == None:
            print ('add_executor', workitem.id, 'not found')
            return
        current_phase.add_executor (resource)
        print(self.name, 'add executor', workitem.id, len(self.phases) - 1, index)

    def remove_executor (self, workitem, resource):
        current_phase, index = self.get_phase(workitem)
        if current_phase == None:
            print('remove_executor', workitem.id, 'not found')
            return
        current_phase.remove_executor (resource)
        print(self.name, 'remove executor', workitem.id, len(self.phases) - 1, index)

    def print_data (self, index):
        if index < len (self.phases):
            self.phases[index].print_data ()

class PipelineManager:
    def __init__ (self, pipelinefile, budget):
        self.pipelinefile = pipelinefile
        self.pipelinestages = []

    def parse_pipelines (self):
        pipelinedatafile = open (self.pipelinefile)
        pipelinedata = yaml.load (pipelinedatafile, Loader = yaml.FullLoader)

        #parse pipeline stages
        index = 0
        for pipelinestage in pipelinedata['pipelinestages']:
            self.pipelinestages.append (PipelineStage(index, pipelinestage))
            index += 1

    def add_workitem_queue (self, workitem, current_time):
        pipelinestage_remove = None
        pipelinestage_add = None

        if workitem == None:
            print ('invalid workitem')
            return
        #print ('add_workitem_queue', workitem.id, workitem.iscomplete)

        if workitem.iscomplete == True:
            if int (workitem.version) >= len (self.pipelinestages) - 1:
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
                if pipelinestage_add.resourcetype != pipelinestage_remove.resourcetype:
                    add_phase = pipelinestage_add.get_phase_index (remove_phase_index)
                    #print ('add_workitem_queue2', add_phase)
                    if add_phase != None:
                        add_phase.add_workitem (workitem, current_time)
                    else:
                        pipelinestage_add.add_workitem_index (remove_phase_index, workitem, current_time)
        else:
            pipelinestage_add = self.pipelinestages[int(workitem.version)]
            #if len (pipelinestage_add.phases) > 0:
            #    print (len (pipelinestage_add.phases), pipelinestage_add.phases[-1].active)
            pipelinestage_add.add_workitem (workitem, current_time)

    def add_executor (self, workitem, resource):
        pipelinestage = self.pipelinestages[int (workitem.version)]
        pipelinestage.add_executor (workitem, resource)

    def remove_executor (self, workitem, resource):
        pipelinestage = self.pipelinestages[int (workitem.version)]
        pipelinestage.remove_executor (workitem, resource)

    def close_phases (self, rmanager, nowtime):
        #print ('close phases')

        cpu_resources = rmanager.get_resources_type ('CPU')
        gpu_resources = rmanager.get_resources_type ('GPU')

        index = 0

        #print ('close phases size', len (self.pipelinestages[0].phases))

        while index < len (self.pipelinestages[0].phases):
            #print ('close phases index', index)
            prev_phase_closed = False

            if index == 0:
                prev_phase_closed = True
            else:
                last_pipelinestage_index = 0
                for pipelinestage in self.pipelinestages:
                    if len (pipelinestage.phases) > index - 1:
                        last_pipelinestage_index += 1
                    else:
                        break

                last_pipelinestage = self.pipelinestages[last_pipelinestage_index - 1]

                #print ('close_phases last stage size', len (last_pipelinestage.phases))

                #print ('close_phases', index, last_pipelinestage.phases[index - 1].pipelinestage, last_pipelinestage.phases[index - 1].active)
                if last_pipelinestage.phases[index - 1].active == False:
                    prev_phase_closed = True

            if prev_phase_closed == False:
                break

            pipelinestage_index = 0

            for pipelinestage in self.pipelinestages:
                if index < len (pipelinestage.phases):
                    phase = pipelinestage.phases[index]
                else:
                    break

                if phase.active == False:
                    pipelinestage_index += 1
                    continue

                if pipelinestage_index == 0:
                    prev_closed = True

                else:
                    prev_pipelinestage = self.pipelinestages[pipelinestage_index - 1]
                    prev_pipelinestage_phase = prev_pipelinestage.phases[index]

                    if prev_pipelinestage_phase.active == True:
                        prev_closed = False
                    else:
                        prev_closed = True

                if prev_closed == False:
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

                    if workitem != None and int(workitem.version) == pipelinestage_index:
                        none_executing = False
                        break

                if none_executing == True:
                    phase.close_phase(nowtime)

                pipelinestage_index += 1

            index += 1

    def print_stage_queue_data (self):
        for index in range (0, len(self.pipelinestages[0].phases)):
            for pipelinestage in self.pipelinestages:
                pipelinestage.print_data (index)

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

    def get_all_pipelinestages (self):
        return self.pipelinestages

    def get_pipelinestages (self, current, resourcetype):
        if current == None:
            if self.pipelinestages[0].get_resourcetype () != resourcetype:
                return None

        elif current.get_resourcetype () == resourcetype or \
            current.get_index () == len (self.pipelinestages) - 1:
            return None

        ret = []

        if current == None:
            index = 0
        else:
            index = current.get_index () + 1

        while index < len (self.pipelinestages) and \
            self.pipelinestages[index].get_resourcetype() == resourcetype:
            ret.append (self.pipelinestages[index])
            index += 1

        return ret

if __name__ == "__main__":
    pipelinefile = sys.argv[1]
    p = PipelineManager(pipelinefile)
    p.parse_pipelines ()
    p.print_data ()
