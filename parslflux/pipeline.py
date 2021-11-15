import yaml
import sys


class Phase:
    def __init__(self, pipelinestage, starttime):
        self.pipelinestage = pipelinestage
        self.starttime = starttime
        self.count = 0
        self.timestamps = {}
        self.endtime = -1
        self.active = True

    def add_workitem (self, currenttime):
        self.count += 1
        self.timestamps[str(currenttime)] = self.count

    def remove_workitem (self, currenttime):
        self.count -= 1
        self.timestamps[str(currenttime)] = self.count

    def set_endtime (self, endtime):
        self.endtime = endtime
        self.active = False

    def print_data (self):
        print (self.pipelinestage, self.timestamps)

class PipelineStageQueue:
    def __init__(self, pipelinestage):
        self.phases = []
        self.pipelinestage = pipelinestage

    def add_phase (self, currenttime):
        new_phase = Phase (self.pipelinestage, currenttime)
        self.phases.append(new_phase)
        return new_phase

    def close_phase (self, currenttime):
        current_phase = self.phases[-1]
        current_phase.set_endtime (currenttime)

    def print_data (self):
        for phase in self.phases:
            print (self.pipelinestage)
            phase.print_data ()

class PipelineStage:
    def __init__ (self, index, pipelinestage):
        self.name = pipelinestage['name']
        self.index = index
        self.resourcetype = pipelinestage['resource']
        self.pipelinestagequeue = PipelineStageQueue(self.name)

    def get_resourcetype (self):
        return self.resourcetype

    def get_index (self):
        return self.index

    def get_name (self):
        return self.name

    def add_workitem (self, current_time):
        print (self.name, len (self.pipelinestagequeue.phases))
        if len (self.pipelinestagequeue.phases) > 0:
            latest_phase = self.pipelinestagequeue.phases[-1]
            if latest_phase.active == False:
                latest_phase = self.pipelinestagequeue.add_phase(current_time)
        else:
            latest_phase = self.pipelinestagequeue.add_phase(current_time)

        latest_phase.add_workitem(current_time)

    def remove_workitem (self, current_time):
        print(self.name, len(self.pipelinestagequeue.phases))
        latest_phase = self.pipelinestagequeue.phases[-1]
        if latest_phase.active == False:
            print ('invalid phase')
        latest_phase.remove_workitem (current_time)

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

        if workitem.iscomplete == True:
            if int (workitem.version) >= len (self.pipelinestages) - 1:
                pipelinestage_remove = self.pipelinestages[int (workitem.version)]
            else:
                pipelinestage_remove = self.pipelinestages[int (workitem.version)]
                pipelinestage_add = self.pipelinestages[int (workitem.version) + 1]

            if pipelinestage_remove != None:
                pipelinestage_remove.remove_workitem (current_time)
            if pipelinestage_add != None:
                pipelinestage_add.add_workitem (current_time)
        else:
            pipelinestage_add = self.pipelinestages[int(workitem.version)]
            pipelinestage_add.add_workitem (current_time)

    def print_data (self):
        for pipelinestage in self.pipelinestages:
            pipelinestage.pipelinestagequeue.print_data ()

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
