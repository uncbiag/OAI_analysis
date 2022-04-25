class PFS:
    def __init__(self, capacity, partitions):
        self.capacity = capacity
        self.partitions = partitions
        self.storage = {}
        self.app_meta = {}
        self.capacity_in_use = 0

        self.single_node_range = [[0, 0.0078125], [0.0078125, 0.03125], [0.03125, 0.125], [0.125, 0.25], [0.25, 0.5],
                                  [0.5, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 8.0], [8.0, 16.0],
                                  [16.0, 32.0], [32.0, 48.0], [48.0, 64.0], [64.0, 80.0], [80.0, 96.0], [96.0, 144.0],
                                  [144.0, 192.0], [192.0, 240.0], [240.0, 288.0], [288.0, 336.0], [336.0, 400.0]]
        self.single_node_bandwidth = [0.09043223418459635, 0.32201812866580487, 1.655767271302073, 2.702753515734608,
                                      4.478009902024657, 6.765439299779977, 8.739885631160844, 10.955016319246072,
                                      12.574994061387583, 12.452561925315276,
                                      13.54950492061699, 13.428340970634844, 13.314094247572147, 13.531242947630885,
                                      12.984826706203691, 13.302306597746554, 13.002179285802002, 12.36111991756025,
                                      13.19023470630156, 13.278076763053424, 13.205042400895945]

    def store_ckpt(self, app_name, ckpt_id, ckpt_size):
        if ckpt_size+self.capacity_in_use <= self.capacity:
            if app_name in self.storage:
                self.storage[app_name].append(ckpt_id)
            else:
                self.storage[app_name] = [ckpt_id]
                self.app_meta[app_name] = ckpt_size
            self.capacity_in_use += ckpt_size
        else:
            if app_name in self.storage:
                self.storage[app_name].pop()
                self.storage[app_name].append(ckpt_id)
            else:
                for k, v in self.storage[app_name]:
                    if self.app_meta[k] > ckpt_size:
                        v.pop()
                        self.capacity_in_use -= self.app_meta[k]
                        break
                self.storage[app_name] = [ckpt_id]
                self.capacity_in_use += ckpt_size

    def get_capacity(self):
        return self.capacity

    def get_max_wrt_thrpt(self):
        return self.max_wrt_thrpt

    def get_max_rd_thrpt(self):
        return self.max_rd_thrpt

    def get_real_wrt_thrpt_single (self, ckpt_size):
        index = 0
        for range in self.single_node_range:
            if range[0] <= ckpt_size and ckpt_size <= range[1]:
                return self.single_node_bandwidth[index]
            index += 1
        return float(self.single_node_bandwidth[-1] * self.single_node_range[-1][1] / ckpt_size);

    def get_real_rd_thrpt_single (self, ckpt_size):
        index = 0
        for range in self.single_node_range:
            if range[0] <= ckpt_size and ckpt_size <= range[1]:
                return self.single_node_bandwidth[index]
            index += 1
        return float(self.single_node_bandwidth[-1] * self.single_node_range[-1][1] / ckpt_size);

    def get_real_wrt_thrpt(self, clients, ckpt_size):
        for memory_range in self.memory_ranges:
            if memory_range.limit >= (ckpt_size / clients):
                return float(memory_range.get_bandwidth(clients, ckpt_size))
        return float (self.memory_ranges[-1].get_bandwidth(clients, ckpt_size))

    def get_real_rd_thrpt(self, clients, ckpt_size):
        for memory_range in self.memory_ranges:
            if memory_range.limit >= (ckpt_size / clients):
                return float(memory_range.get_bandwidth(clients, ckpt_size))
        return float(self.memory_ranges[-1].get_bandwidth(clients, ckpt_size))

    def search_ckpt(self, app_name, ckpt_id):
        if app_name in self.storage:
            if ckpt_id in self.storage[app_name]:
                return True
            else:
                return False
        else:
            return False

    def get_free_capacity(self):
        return self.capacity-self.capacity_in_use