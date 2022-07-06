from parslfluxsim.workitem_sim import WorkItem
import copy

class BagOfWorkItems (object):
    def __init__(self, pipelinestageindex, computetype, env):
        self.pipelinestageindex = pipelinestageindex
        self.bag = []
        self.computetype = computetype
        self.root = True
        self.env = env
        self.snapshots = {}

    def set_root (self, status):
        self.root = status

    def add_workitem (self, workitem):
        self.bag.append(workitem)
        self.snapshots[str (self.env.now)] = len (self.bag)

    def pop_workitem (self):
        if len (self.bag) > 0:
            item = self.bag.pop(0)
            self.snapshots[str(self.env.now)] = len(self.bag)
            return item
        return None

    def sort_bag_by_earliest_schedule_time (self):
        self.bag = sorted (self.bag, key=lambda x:x.scheduletime)

    def sort_complete_workitems_by_priority (self):
        self.bag = sorted(self.bag, key=lambda x: (x.priority), reverse=True)

    def sort_complete_workitems_by_stage (self):
        self.bag = sorted(self.bag, key=lambda x: (x.version), reverse=True)

    def sort_complete_workitems_by_earliest_finish_time (self):
        self.bag = sorted (self.bag, key=lambda x:x.endtime)

    def sort_complete_workitems_by_latest_finish_time (self):
        self.bag = sorted (self.bag, key=lambda x:x.endtime, reverse=True)

    def sort_complete_workitems_by_transfer_latency (self, domain_id):
        if self.root == True:
            return
        self.bag = sorted (self.bag, key=lambda x:x.input_transfer_latency_map[str(domain_id)])