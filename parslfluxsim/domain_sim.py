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

    def get_pfs_handle(self):
        return self.pfs

    def reset (self):
        self.pfs = PFS(1048576, self.env)

class HPCDomain (Domain):
    def init_cluster(self):
        self.type = 'HPC'
        for cputype in self.resourcedata['CPU']:
            self.resourcedict[cputype['id']] = {}
            self.resourcedict[cputype['id']]['computetype'] = 'CPU'
            self.resourcedict[cputype['id']]['cost'] = cputype['cost']
            self.resourcedict[cputype['id']]['startuptime'] = cputype['startuptime']

        for gputype in self.resourcedata['GPU']:
            self.resourcedict[gputype['id']] = {}
            self.resourcedict[gputype['id']]['computetype'] = 'GPU'
            self.resourcedict[gputype['id']]['cost'] = gputype['cost']
            self.resourcedict[gputype['id']]['startuptime'] = gputype['startuptime']

        self.performancedata = read_performance_data()
        self.pfs = PFS(1048576, self.env)

    def provision_resource (self, resourcetype, on_demand, bidding_price):
        return 0, 0


class AWSDomain (Domain):

    def provision_on_demand_resource (self, resourcetype):

        cost = self.reservationcostmodel.get_on_demand_cost (resourcetype)

        provision_time = self.reservationcostmodel.get_on_demand_startup_time (resourcetype)

        print (cost, provision_time)

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
        self.type = 'AWS'

        for cputype in self.resourcedata['CPU']:
            self.resourcedict[cputype['id']] = {}
            self.resourcedict[cputype['id']]['computetype'] = 'CPU'
            self.resourcedict[cputype['id']]['cost'] = cputype['cost']
            self.resourcedict[cputype['id']]['startuptime'] = cputype['startuptime']

        for gputype in self.resourcedata['GPU']:
            self.resourcedict[gputype['id']] = {}
            self.resourcedict[gputype['id']]['computetype'] = 'GPU'
            self.resourcedict[gputype['id']]['cost'] = gputype['cost']
            self.resourcedict[gputype['id']]['startuptime'] = gputype['startuptime']


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

    def get_central_repository_latency (self, size):
        return float (size / 1024)

    def reset (self):
        for domain in self.domains:
            self.domains[domain].reset()

    def get_interdomain_transfer_latency (self, src_domain_id, dest_domain_id, filesize):
        return filesize / self.interdomain_transfer_rate[str(src_domain_id)][str(dest_domain_id)]

    def get_domains (self):
        return list (self.domains.values())

    def get_domain (self, id):
        if id in self.domains.keys ():
            return self.domains[str (id)]
        else:
            return None

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
                self.interdomain_transfer_rate[str(src_domain_id)][str(dst_domain_id)] = 1

if __name__ == '__main__':
    dmanager = DomainManager ('md_resources.yml')
    dmanager.init_resource_model()