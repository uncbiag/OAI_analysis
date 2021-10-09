import yaml
import sys

class PipelineStage:
    def __init__ (self, index, pipelinestage):
        self.name = pipelinestage['name']
        self.index = index
        self.resourcetype = pipelinestage['resource']

    def get_resourcetype (self):
        return self.resourcetype

    def get_index (self):
        return self.index

    def get_name (self):
        return self.name

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
