import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import datetime
import statistics
import csv
from matplotlib.lines import Line2D
import pandas as pd

import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
from scipy.stats import *
import pickle

cpu_completion_files = [
    'plots/run_64_100_77_98_First_0/complete.txt',
    'plots/run_65_100_99_94_First_0/complete.txt',
    'plots/run_66_100_44_93_First_0/complete.txt',
    'plots/run_67_100_93_44_First_0/complete.txt',
    'plots/run_69_100_98_94_First_0/complete.txt'
]

gpu_completion_files = [
    'plots/run_64_100_77_98_First_0/complete.txt',
    'plots/run_67_100_93_44_First_0/complete.txt',
    'plots/run_68_100_99_77_First_0/complete.txt'
]

cpu_resource_types = {
    'c77': 'Epyc Rome', #C5a
    'c99': 'Intel Ivy Bridge',
    'c44': 'AMD Opteron',
    'c93': 'Intel Broadwell',
    'c98': 'Intel Sandy Bridge'
}

gpu_resource_types = {
    'c44': 'Nvidia GTX 1080',
    'c77': 'Nvidia RTX 2060 Super',
    'c93': 'Nvidia RTX 2070',
    'c94': 'Nvidia RTX 2070',
    'c98': 'Nvidia RTX 2070'
}

def read_performance_data ():

    results = {}

    resource_data = {}

    for cpu_completion_file in cpu_completion_files:
        completion_file = open(cpu_completion_file, "r")
        cpu_completion_lines = completion_file.readlines ()
        images = []

        min_start_time = None

        for cpu_completion_line in cpu_completion_lines:
            _, _, imageid, version, _, _, startdate, starthour, enddate, endhour, resourceid, _ = cpu_completion_line.split(' ',
                                                                                                                    11)
            if int(version) % 2 == 1:
                continue

            starttime_s = startdate + ' ' + starthour
            endtime_s = enddate + ' ' + endhour

            starttime = datetime.datetime.strptime(starttime_s,
                                                   '%Y-%m-%d %H:%M:%S').timestamp()
            endtime = datetime.datetime.strptime(endtime_s, '%Y-%m-%d %H:%M:%S').timestamp()

            if min_start_time == None:
                min_start_time = starttime
            elif min_start_time > starttime:
                min_start_time = starttime


            resourceid = resourceid.strip()
            images.append([imageid, version, cpu_resource_types[resourceid], starttime, endtime])

        for image in images:
            image[3] -= min_start_time
            image[4] -= min_start_time

        for image in images:
            imageid = image[0]
            version = image[1]
            resourceid = image[2]
            starttime = image[3]
            endtime = image[4]

            if resourceid not in resource_data:
                resource_data[resourceid] = {}
                if version not in resource_data[resourceid]:
                    resource_data[resourceid][version] = []
                    resource_data[resourceid][version].append (endtime - starttime)
                else:
                    resource_data[resourceid][version].append(endtime - starttime)
            else:
                if version not in resource_data[resourceid]:
                    resource_data[resourceid][version] = []
                    resource_data[resourceid][version].append (endtime - starttime)
                else:
                    resource_data[resourceid][version].append(endtime - starttime)

    for resourceid in resource_data.keys():
        for version in resource_data[resourceid].keys():
            csv_filename = resourceid + version + '.csv'
            data = {version: resource_data[resourceid][version]}
            #print (resourceid, version)

            shape, location, scale = gamma.fit(resource_data[resourceid][version])
            dist = 'gamma'
            #print(resourceid, version, shape, location, scale)
            if resourceid not in results:
                results[resourceid] = {}
            results[resourceid][version] = [dist, shape, location, scale]

    resource_data = {}

    for gpu_completion_file in gpu_completion_files:
        completion_file = open(gpu_completion_file, "r")
        gpu_completion_lines = completion_file.readlines ()
        images = []

        min_start_time = None

        for gpu_completion_line in gpu_completion_lines:
            _, _, imageid, version, _, _, startdate, starthour, enddate, endhour, resourceid, _ = gpu_completion_line.split(' ',
                                                                                                                    11)
            if int(version) % 2 == 0:
                continue

            starttime_s = startdate + ' ' + starthour
            endtime_s = enddate + ' ' + endhour

            starttime = datetime.datetime.strptime(starttime_s,
                                                   '%Y-%m-%d %H:%M:%S').timestamp()
            endtime = datetime.datetime.strptime(endtime_s, '%Y-%m-%d %H:%M:%S').timestamp()

            if min_start_time == None:
                min_start_time = starttime
            elif min_start_time > starttime:
                min_start_time = starttime


            resourceid = resourceid.strip()
            images.append([imageid, version, gpu_resource_types[resourceid], starttime, endtime])

        for image in images:
            image[3] -= min_start_time
            image[4] -= min_start_time

        for image in images:
            imageid = image[0]
            version = image[1]
            resourceid = image[2]
            starttime = image[3]
            endtime = image[4]

            if resourceid not in resource_data:
                resource_data[resourceid] = {}
                if version not in resource_data[resourceid]:
                    resource_data[resourceid][version] = []
                    resource_data[resourceid][version].append (endtime - starttime)
                else:
                    resource_data[resourceid][version].append(endtime - starttime)
            else:
                if version not in resource_data[resourceid]:
                    resource_data[resourceid][version] = []
                    resource_data[resourceid][version].append (endtime - starttime)
                else:
                    resource_data[resourceid][version].append(endtime - starttime)

    for resourceid in resource_data.keys():
        for version in resource_data[resourceid].keys():
            csv_filename = resourceid + version + '.csv'
            data = {version: resource_data[resourceid][version]}
            #print (resourceid, version)

            if version == '1':
                shape, location, scale = lognorm.fit(resource_data[resourceid][version])
                dist = 'lognorm'
            else:
                shape, location, scale = gamma.fit(resource_data[resourceid][version])
                dist = 'gamma'
            #print(resourceid, version, shape, location, scale)
            if resourceid not in results:
                results[resourceid] = {}
            results[resourceid][version] = [dist, shape, location, scale]


    print (results)
    return results