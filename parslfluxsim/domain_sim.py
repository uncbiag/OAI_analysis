import yaml
from aws_cloud_sim.costmodel import AWSCostModel
from parslfluxsim.performance_sim import read_performance_data
from aws_cloud_sim.arc_to_aws_mapping import ARCMapping
from filesystem.pfs import PFS

class Domain:
    def __init__(self, resourcedata, domain_id, env):
        self.resourcedict = {}
        self.datacostdict = {}
        self.name = resourcedata['name']
        self.resourcedata = resourcedata
        self.id = domain_id
        self.env = env
        self.reservation_quota = 0
        self.allocations = {}

    def print_data (self):
        print ('resource dict', self.resourcedict)
        print ('cost_dict', self.datacostdict)

    def get_resource_types (self, computetype):
        results = []
        #print (self.resourcedict)
        for resource_key in self.resourcedict.keys():
            if self.resourcedict[resource_key]['computetype'] == computetype:
                results.append (resource_key)

        return results

    def get_performance_dist (self, resourcetype):
        if resourcetype in self.performancedata:
            return self.performancedata[resourcetype]

        return None

    def provision_resource (self, resourcetype, on_demand, bidding_price):
        pass

    def get_allocations (self):
        return self.allocations

    def add_allocation (self, resourcetype):
        if resourcetype not in self.allocations:
            self.allocations[resourcetype] = 0
        self.allocations[resourcetype] += 1

    def remove_allocation (self, resourcetype):
        if resourcetype not in self.allocations or self.allocations[resourcetype] <= 0:
            print ('remove_allocation ()', resourcetype, 'not allocated')
            return

        self.allocations[resourcetype] -= 1

    def get_reservation_quota (self):
        return self.reservation_quota

    def get_availability (self, resourcetype, count):
        pass

    def get_pfs_handle(self):
        return self.pfs

    def reset (self):
        self.pfs.reset()

class HPCDomain (Domain):
    def init_cluster(self):
        self.type = 'HPC'
        self.reservation_quota = self.resourcedata['reservation_quota']
        for cputype in self.resourcedata['CPU']:
            self.resourcedict[cputype['id']] = {}
            self.resourcedict[cputype['id']]['computetype'] = 'CPU'
            self.resourcedict[cputype['id']]['cost'] = cputype['cost']
            self.resourcedict[cputype['id']]['startuptime'] = cputype['startuptime']
            self.resourcedict[cputype['id']]['limit'] = cputype['limit']

        for gputype in self.resourcedata['GPU']:
            self.resourcedict[gputype['id']] = {}
            self.resourcedict[gputype['id']]['computetype'] = 'GPU'
            self.resourcedict[gputype['id']]['cost'] = gputype['cost']
            self.resourcedict[gputype['id']]['startuptime'] = gputype['startuptime']
            self.resourcedict[gputype['id']]['limit'] = gputype['limit']

        self.performancedata = read_performance_data()
        self.pfs = PFS(1048576, self.env)

    def provision_resource (self, resourcetype, on_demand, bidding_price):
        #print (resourcetype)
        #print (self.resourcedict.keys())
        if self.resourcedict[resourcetype]['limit'] != -1 and resourcetype in self.allocations.keys():
            if self.allocations[resourcetype] < self.resourcedict[resourcetype]['limit']:
                #print ('ARC provision resource ()', self.allocations[resourcetype], self.resourcedict[resourcetype]['limit'])
                return 0, 0
            else:
                return None, None
        else:
            return 0, 0

    def get_availability(self, resourcetype, count):
        if self.resourcedict[resourcetype]['limit'] != -1 and resourcetype in self.allocations.keys():
            if self.allocations[resourcetype] + count < self.resourcedict[resourcetype]['limit']:
                return True
            else:
                return False
        else:
            return True


class AWSDomain (Domain):

    def get_availability(self, resourcetype, count):
        return True

    def provision_on_demand_resource (self, resourcetype):

        cost = self.reservationcostmodel.get_on_demand_cost (resourcetype)

        provision_time = self.reservationcostmodel.get_on_demand_startup_time (resourcetype)

        return cost, provision_time

    def provision_spot_resource (self, resourcetype, bidding_price):

        cost =  self.reservationcostmodel.get_spot_cost(resourcetype, bidding_price)

        provision_time = self.reservationcostmodel.get_spot_startup_time(resourcetype)

        return cost, provision_time

    def provision_resource (self, resourcetype, on_demand, bidding_price):
        if on_demand == True:
            return self.provision_on_demand_resource (resourcetype)
        else:
            return self.provision_spot_resource (resourcetype, bidding_price)

    def init_cluster (self):
        self.type = 'CLOUD'
        self.reservation_quota = self.resourcedata['reservation_quota']

        for cputype in self.resourcedata['CPU']:
            self.resourcedict[cputype['id']] = {}
            self.resourcedict[cputype['id']]['computetype'] = 'CPU'
            self.resourcedict[cputype['id']]['cost'] = cputype['cost']
            self.resourcedict[cputype['id']]['startuptime'] = cputype['startuptime']
            self.resourcedict[cputype['id']]['limit'] = cputype['limit']

        for gputype in self.resourcedata['GPU']:
            self.resourcedict[gputype['id']] = {}
            self.resourcedict[gputype['id']]['computetype'] = 'GPU'
            self.resourcedict[gputype['id']]['cost'] = gputype['cost']
            self.resourcedict[gputype['id']]['startuptime'] = gputype['startuptime']
            self.resourcedict[gputype['id']]['limit'] = gputype['limit']


        self.datacostdict['storage'] = {}
        self.datacostdict['storage']['volume'] = self.resourcedata['COST']['storage']['volume']
        self.datacostdict['storage']['iops'] = self.resourcedata['COST']['storage']['iops']
        self.datacostdict['storage']['performance'] = self.resourcedata['COST']['storage']['performance']

        self.datacostdict['datamovement'] = {}
        self.datacostdict['datamovement']['in'] = self.resourcedata['COST']['datamovement']['in']
        self.datacostdict['datamovement']['out'] = {}
        self.datacostdict['datamovement']['out']['internet'] = self.resourcedata['COST']['datamovement']['out']['internet']
        self.datacostdict['datamovement']['out']['inter-domain'] = self.resourcedata['COST']['datamovement']['out']['inter-domain']

        self.reservationcostmodel = AWSCostModel('us-east-1a', 'Linux/UNIX', self.env)

        self.mapping = ARCMapping()

        self.performancedata = read_performance_data()
        # replace ARC resources with AWS
        self.performancedata = self.mapping.replace_arc_with_aws(self.performancedata)
        self.resourcedict = self.mapping.replace_arc_with_aws(self.resourcedict)

        self.pfs = PFS(1048576, self.env)


class DomainManager:
    def __init__(self, domainfile, env):
        self.domainfile = domainfile
        self.domains = {}
        self.env = env
        self.interdomain_transfer_rate = {}
        self.interdomain_transfer_history = {}

    def get_data_transer_history (self):
        return self.interdomain_transfer_history

    def add_data_transfer (self, src_domain_id, dest_domain_id, transfer_size, pipelinestage):
        if str(src_domain_id) not in self.interdomain_transfer_history:
            self.interdomain_transfer_history[str(src_domain_id)] = {}

        if str(dest_domain_id) not in self.interdomain_transfer_history[str(src_domain_id)]:
            self.interdomain_transfer_history[str(src_domain_id)][str(dest_domain_id)] = {}

        if pipelinestage.name not in self.interdomain_transfer_history[str(src_domain_id)][str(dest_domain_id)]:
            self.interdomain_transfer_history[str(src_domain_id)][str(dest_domain_id)][pipelinestage.name] = []

        if len (self.interdomain_transfer_history[str(src_domain_id)][str(dest_domain_id)][pipelinestage.name]) <= 0:
            self.interdomain_transfer_history[str(src_domain_id)][str(dest_domain_id)][pipelinestage.name].append ([self.env.now, transfer_size])
        else:
            till_now = self.interdomain_transfer_history[str(src_domain_id)][str(dest_domain_id)][pipelinestage.name][-1][1]
            self.interdomain_transfer_history[str(src_domain_id)][str(dest_domain_id)][pipelinestage.name].append ([self.env.now, transfer_size + till_now])

    def get_central_repository_latency (self, size):
        return float (size / 1024)

    def reset (self):
        self.reset_time = self.env.now
        for domain in self.domains:
            self.domains[domain].reset()

        self.interdomain_transfer_history = {}

    def get_interdomain_transfer_latency (self, src_domain_id, dest_domain_id, filesize):
        if str(src_domain_id) == str(dest_domain_id):
            return 0
        return filesize / self.interdomain_transfer_rate[str(src_domain_id)][str(dest_domain_id)]

    def get_domains (self):
        return list (self.domains.values())

    def get_domain (self, id):
        if id in self.domains.keys ():
            return self.domains[str (id)]
        else:
            return None

    def get_hpc_domains (self):
        results = []
        for key in self.domains.keys ():
            domain = self.domains[key]
            if domain.type == 'HPC':
                results.append(domain)
        return results

    def get_cloud_domains (self):
        results = []
        for key in self.domains.keys():
            domain = self.domains[key]
            if domain.type == 'CLOUD':
                results.append(domain)
        return results

    def init_resource_model (self):
        yaml_resourcefile = open(self.domainfile)
        domain_yaml = yaml.load(yaml_resourcefile, Loader=yaml.FullLoader)

        domain_id = 0
        for domain in domain_yaml['domains']:
            if domain['type'] == 'HPC':
                domain_object = HPCDomain (domain, domain_id, self.env)
            elif domain['type'] == 'AWS':
                domain_object = AWSDomain (domain, domain_id, self.env)

            domain_object.init_cluster ()
            self.domains[str (domain_id)] = domain_object
            domain_object.print_data()
            domain_id += 1

        for src_domain_id in self.domains.keys():
            self.interdomain_transfer_rate[str(src_domain_id)] = {}
            for dst_domain_id in self.domains.keys ():
                if src_domain_id == dst_domain_id:
                    continue
                self.interdomain_transfer_rate[str(src_domain_id)][str(dst_domain_id)] = 1024

if __name__ == '__main__':
    dmanager = DomainManager ('md_resources.yml')
    dmanager.init_resource_model()