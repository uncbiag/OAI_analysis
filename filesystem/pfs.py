
class PFS:
    def __init__(self, capacity, env):
        self.capacity = capacity
        self.storage = {}
        self.deleted_entries = {}
        self.capacity_in_use = 0
        self.env = env
        self.total_entries = 0
        self.read_bandwidth = 1024
        self.write_bandwidth = 1024

    def store_file (self, workitem, children_pipelinestages):
        image_id = workitem.id
        pipelinestagename = workitem.pipelinestage.name
        filesize = workitem.pipelinestage.output_size

        if str (pipelinestagename) not in self.storage.keys ():
            self.storage [str(pipelinestagename)] = {}

        if image_id not in self.storage[str(pipelinestagename)]:
            self.storage[str(pipelinestagename)][str(image_id)] = {}
            self.storage[str(pipelinestagename)][str(image_id)]['entrytime'] = workitem.output_write_endtime
            self.storage[str(pipelinestagename)][str(image_id)]['size'] = filesize

            self.capacity_in_use += filesize
            self.total_entries += 1

            self.storage[str(pipelinestagename)][str(image_id)]['pending_children_read'] = {}

            for child_pipelinestage in children_pipelinestages:
                self.storage[str(pipelinestagename)][str(image_id)]['pending_children_read'][str(child_pipelinestage.name)] = False

            #print ('pfs store ()', image_id, pipelinestagename, filesize, self.storage[str(pipelinestagename)][str(image_id)]['pending_children_read'])

    def read_file (self, image_id, current_pipelinestagename, parent_pipelinestagename):
        if str(parent_pipelinestagename) not in self.storage.keys ():
            print ('read_file () 1', current_pipelinestagename, parent_pipelinestagename)
            return None

        if str(image_id) not in self.storage[str(parent_pipelinestagename)].keys ():
            print('read_file () 1', parent_pipelinestagename, str(image_id))
            return None

        #print('read_file ()', self.storage[str(parent_pipelinestagename)][str(image_id)])

        #self.storage[str(parent_pipelinestagename)][str(image_id)]['pending_children_read'].remove (str(current_pipelinestagename))

        read_latency = self.storage[str(parent_pipelinestagename)][str(image_id)]['size'] / self.read_bandwidth
        read_time = self.env.now + read_latency

        if 'latest_read_time' not in self.storage[str(parent_pipelinestagename)][str(image_id)].keys ():
            self.storage[str(parent_pipelinestagename)][str(image_id)]['latest_read_time'] = read_time
        else:
            if self.storage[str(parent_pipelinestagename)][str(image_id)]['latest_read_time'] < read_time:
                self.storage[str(parent_pipelinestagename)][str(image_id)]['latest_read_time'] = read_time

        #if len (self.storage[str(parent_pipelinestagename)][str(image_id)]['pending_children_read']) <= 0:
        #    self.delete_file (image_id, parent_pipelinestagename)

        return read_latency

    def delete_file (self, image_id, parent_pipelinestagename, current_pipelinestagename):
        #print('delete_file ()', image_id, parent_pipelinestagename)
        self.storage[str(parent_pipelinestagename)][str(image_id)]['pending_children_read'][str(current_pipelinestagename)] = True

        pending_children_reads = self.storage[str(parent_pipelinestagename)][str(image_id)]['pending_children_read']

        for pipelinestage in pending_children_reads.keys ():
            if pending_children_reads[pipelinestage] == False:
                return

        filesize = self.storage[str(parent_pipelinestagename)][str (image_id)]['size']
        entrytime = self.storage[str(parent_pipelinestagename)][str (image_id)]['entrytime']
        exittime = self.storage[str(parent_pipelinestagename)][str (image_id)]['latest_read_time']

        if str (parent_pipelinestagename) not in self.deleted_entries:
            self.deleted_entries[str(parent_pipelinestagename)] = {}
        self.deleted_entries[str(parent_pipelinestagename)][str(image_id)] = {}
        self.deleted_entries[str(parent_pipelinestagename)][str(image_id)]['entry'] = entrytime
        self.deleted_entries[str(parent_pipelinestagename)][str(image_id)]['exit'] = exittime
        self.deleted_entries[str(parent_pipelinestagename)][str(image_id)]['size'] = filesize


        del self.storage[str(parent_pipelinestagename)][str(image_id)]
        self.capacity_in_use -= filesize
        self.total_entries -= 1


    def get_read_latency (self, size):
        return float (size/self.read_bandwidth)

    def get_write_latency (self, size):
        return float(size / self.write_bandwidth)

    def get_delete_entries (self):
        return self.deleted_entries

    def get_capacity(self):
        return self.capacity

    def get_free_capacity(self):
        return self.capacity-self.capacity_in_use

    def print_data (self):
        print (len (list (self.storage.keys())))
        for key in self.storage:
            print (key, len (list (self.storage[key].keys())))