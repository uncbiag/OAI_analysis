from parsl.config import Config
from parsl.channels import SSHChannel
from parsl.channels import LocalChannel
from parsl.providers import SlurmProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_query
from parsl import python_app, bash_app


import parsl, os

import platform
from parslflux.resources import ResourceManager

def get_launch_config (options):
    inspector_config = Config (
        executors=[
            HighThroughputExecutor(
                label="inspector",
                address=str(platform.node().split('.')[0]),
                max_workers=1,
                provider=SlurmProvider(
                    channel=LocalChannel(),
                    nodes_per_block=1,
                    min_blocks=1,
                    max_blocks=1,
                    init_blocks=1,
                    partition='normal',
                    walltime='24:00:00',
                    scheduler_options=options,
                    launcher=SrunLauncher(),
                ),
            )
        ],
        app_cache=False,
    )
    return inspector_config

def get_own_remote_uri():
    localuri = os.getenv('FLUX_URI')
    remoteuri = localuri.replace("local://", "ssh://" + platform.node().split('.')[0])
    return remoteuri

@bash_app
def app (command):
    print (command)
    return ''+command

@bash_app
def app1 (entity, entityvalue, workerid, scheduleruri):
    return '~/local/bin/flux start -o,--setattr=entity={},--setattr=entityvalue={} python3.6 flux_wrapper.py {} {}'.format('TASK', entityvalue, workerid, scheduleruri)

def launch_worker (resource):
    parsl.clear()
    options = '#SBATCH -w ' + resource.hostname
    config = get_launch_config (options)
    parsl.load (config)
    app1 ('TASK', resource.hostname, resource.hostname, get_own_remote_uri ())
   # app ('~/local/bin/flux start -o,--setattr=entity={},--setattr=entityvalue={} python3.6 flux_wrapper.py {} {}'.format('TASK', resource.hostname, resource.hostname, get_own_remote_uri()))
    print ('launched', resource.hostname)

def request_reserved_worker ():
    rmanager = get_resource_manager ()
    resource = rmanager.request_reserved_resource ()
    if resource == None:
        print ('no more reserved resources')
    else:
        launch_worker (resource)
