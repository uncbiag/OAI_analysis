from parslfluxsim.workitem_sim import WorkItem
import copy

class BagOfWorkItems (object):
    def __init__(self, pipelinestageindex, computetype):
        self.pipelinestageindex = pipelinestageindex
        self.bag = []
        self.computetype = computetype

    def add_workitem (self, workitem):
        self.bag.append(workitem)

    def pop_workitem (self):
        if len (self.bag) > 0:
            item = self.bag.pop(0)
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