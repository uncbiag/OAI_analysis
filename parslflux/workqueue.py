
class WorkItemQueue:
    def __init__ (self):
        self.queue = []

    def get_workitem (self):
        if len (self.queue) == 0:
            return None

        return self.queue[0]

    def pop_workitem (self):
        if len (self.queue) == 0:
            return None

        return self.queue.pop (0)

    def is_empty (self):
        if len (self.queue) == 0:
            return True

        return  False

    def add_workitem (self, workitem):
        self.queue.append (workitem)
