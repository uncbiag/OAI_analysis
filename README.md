About:
-----
The aim of this project is to optimize and to openly provide osteoarthritis community a new technology to rapidly and automatically measure cartilage thickness, appearance and changes on magnetic resonance images (MRI) of the knee for huge image databases. This will allow assessment of trajectories of cartilage loss over time and associations with clinical outcomes on an unprecedented scale; future work will focus on incorporating additional disease markers, ranging from MRI-derived biomarkers for bone and synovial lesions, to biochemical biomarkers, to genetic information. novel, efficiently parallelized codes and tools for pairwise, group-wise and longitudinal analysis of high-resolution MR images suitable for large image repositories with the potential to produce highly reproducible diagnostic and monitoring capabilities by exploiting massively parallel computing on an HPC cluster and cloud facilities.

Cluster Setup:
------------
To setup the AWS cluster, follow the below link.

https://d1.awsstatic.com/Projects/P4114756/deploy-elastic-hpc-cluster_project.pdf

The HPC cluster can be setup using amazon's parallelcluster tool as explained in the above link.
Below is one example of the parallelcluster config that was used for OAI's cluster setup and experiments.

[aws]
aws_region_name = us-east-1

[aliases]
ssh = ssh {CFN_USER}@{MASTER_IP} {ARGS}

[global]
cluster_template = default
update_check = true
sanity_check = true

[cluster default]
key_name = ssb_ncsu
base_os = centos7
scheduler = slurm
master_instance_type = t2.large
vpc_settings = default
queue_settings = compute

[vpc default]
vpc_id = vpc-0504e2077be140d92
master_subnet_id = subnet-0fe47aed0580168aa
compute_subnet_id = subnet-0273b0759c3100949
use_public_ips = false

[queue compute]
enable_efa = false
compute_resource_settings = default

[compute_resource default]
instance_type = g4dn.xlarge
min_count = 2
max_count = 2

Installation:
------------

Automated:
---------
Once cluster is setup, run the setup.sh script. It will install the dependency packages on master node and compute nodes.

Manual:
------
Read the setup.sh script and install each package manually.
setup.sh installs nifty-reg. To install easyreg, follow the below procedure.


Easyreg installation (requires python >=3.7):
-------------------------------------------

cd oai
git clone https://github.com/uncbiag/easyreg.git
cd easyreg
pip install -r requirements.txt
# Download the pretrained model (in mermaid directory) (a seven-step affine network with a three-step vSVF model)
cd demo && mkdir pretrained && cd pretrained
gdown https://drive.google.com/uc?id=1f7pWcwGPvr28u4rr3dAL_4aD98B7PNDX
cd ../..
git clone https://github.com/uncbiag/mermaid.git
cd mermaid
python setup.py develop


Getting the Data:
----------------
Download the data from https://nda.nih.gov/oai/accessing_images.html

Running Experiments:
-------------------
You can configure the data and other application paths in oai_analysis_longleaf.json.
You can either use the run.batch or run_analysis_on_slurm_cluster.sh script to submit
batch jobs to slurm.
