
class PFS:
    def __init__(self, capacity, env):
        self.capacity = capacity
        self.storage = {}
        self.deleted_entries = {}
        self.capacity_in_use = 0
        self.env = env
        self.total_entries = 0

    def store_file (self, workitem, children_pipelinestages):
        image_id = workitem.id
        version = workitem.version
        filesize = workitem.pipelinestage.output_size

        if str (version) not in self.storage.keys ():
            self.storage [str(version)] = {}
        if image_id not in self.storage[str(version)]:
            self.storage[str(version)][str(image_id)] = {}
            self.storage[str(version)][str(image_id)]['entrytime'] = workitem.output_write_endtime
            self.storage[str(version)][str(image_id)]['size'] = filesize

            self.capacity_in_use += filesize
            self.total_entries += 1
            pipelinestages = []
            for child_pipelinestage in children_pipelinestages:
                pipelinestages.append (child_pipelinestage.index)

            self.storage[str(version)][str(image_id)]['children_status'] = pipelinestages

    def delete_file (self, workitem, current_pipelinestageindex, parent_pipelinestageindex):
        image_id = workitem.id
        version = parent_pipelinestageindex

        if str(version) not in self.storage.keys():
            print ('pfs ()', 'delete_file ()', version, 'not present')
            return

        if str(image_id) not in self.storage[str(version)].keys():
            print ('pfs ()', 'delete_file ()', image_id, 'not present')
            return

        if current_pipelinestageindex not in self.storage[str(version)][str(image_id)]['children_status']:
            print ('pfs ()', 'delete_file ()', current_pipelinestageindex, 'not present')
            return

        self.storage[str(version)][str (image_id)]['children_status'].remove (current_pipelinestageindex)

        if 'latest_removal_time' not in self.storage[str(version)][str(image_id)]:
            self.storage[str(version)][str(image_id)]['latest_removal_time'] = workitem.input_read_endtime
        else:
            if workitem.input_read_endtime > self.storage[str(version)][str(image_id)]['latest_removal_time']:
                self.storage[str(version)][str(image_id)]['latest_removal_time'] = workitem.input_read_endtime


        if len (self.storage[str(version)][str (image_id)]['children_status']) <= 0:
            filesize = self.storage[str(version)][str (image_id)]['size']
            entrytime = self.storage[str(version)][str (image_id)]['entrytime']
            exittime = self.storage[str(version)][str (image_id)]['latest_removal_time']

            if str (version) not in self.deleted_entries:
                self.deleted_entries[str(version)] = {}
                self.deleted_entries[str(version)][str(image_id)] = {}
                self.deleted_entries[str(version)][str(image_id)]['entry'] = entrytime
                self.deleted_entries[str(version)][str(image_id)]['exit'] = exittime


            del self.storage[str(version)][str(image_id)]
            self.capacity_in_use -= filesize
            self.total_entries -= 1

    def get_data (self):
        return self.deleted_entries

    def get_capacity(self):
        return self.capacity

    def serarch_image(self, image_id, version):
        if version in self.storage:
            if image_id in self.storage[version]:
                return True
            else:
                return False
        else:
            return False

    def get_free_capacity(self):
        return self.capacity-self.capacity_in_use

    def print_data (self):
        print (len (list (self.storage.keys())))
        for key in self.storage:
            print (key, len (list (self.storage[key].keys())))