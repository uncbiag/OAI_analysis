
#'https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/compute-optimized-instances.html'
#'https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/accelerated-computing-instances.html'
#'https://instances.vantage.sh/'
class ARCMapping:
    def __init__(self):
        self.arctoaws = {}
        self.awstoarc = {}
        self.map_arc_to_aws()
        self.map_aws_to_arc()

    def map_arc_to_aws (self):
        self.arctoaws['Epyc Rome'] = 'c5a.4xlarge'
        self.arctoaws['Intel Ivy Bridge'] = 'c5n.4xlarge'
        self.arctoaws['AMD Opteron'] = 'c4.4xlarge'
        self.arctoaws['Intel Broadwell'] = 'c5.4xlarge'
        self.arctoaws['Intel Sandy Bridge'] = 'c6i.4xlarge'
        self.arctoaws['Nvidia GTX 1080'] = 'g3.4xlarge'
        self.arctoaws['Nvidia RTX 2060 Super'] = 'g4ad.4xlarge'
        self.arctoaws['Nvidia RTX 2070'] = 'g5.xlarge'

    def map_aws_to_arc (self):
        self.awstoarc['c5a.4xlarge'] = 'Epyc Rome'
        self.awstoarc['c5n.4xlarge'] = 'Intel Ivy Bridge'
        self.awstoarc['c4.4xlarge'] = 'AMD Opteron'
        self.awstoarc['c5.4xlarge'] = 'Intel Broadwell'
        self.awstoarc['c6i.4xlarge'] = 'Intel Sandy Bridge'
        self.awstoarc['g3.4xlarge'] = 'Nvidia GTX 1080'
        self.awstoarc['g4ad.4xlarge'] = 'Nvidia RTX 2060 Super'
        self.awstoarc['g5.xlarge'] = 'Nvidia RTX 2070'

    def replace_arc_with_aws (self, mapping):
        results = {}
        for key in mapping:
            results[self.arctoaws[key]] = mapping[key]

        print (mapping)
        print (results)
        return results

    def get_arc_to_aws (self, resourcetype):
        return self.arctoaws[resourcetype]
    def get_aws_to_arc (self, resourcetype):
        return self.awstoarc[resourcetype]