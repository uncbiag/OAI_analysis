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


matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['font.family'] = 'Times New Roman'

if __name__ == '__main__':
    cpu_completion_files = [
        'run_64_100_77_98_First_0/complete.txt',
        'run_65_100_99_94_First_0/complete.txt',
        'run_66_100_44_93_First_0/complete.txt',
        'run_67_100_93_44_First_0/complete.txt',
        'run_69_100_98_94_First_0/complete.txt'
    ]

    gpu_completion_files = [
        'run_64_100_77_98_First_0/complete.txt',
        'run_67_100_93_44_First_0/complete.txt',
        'run_68_100_99_77_First_0/complete.txt'
    ]

    cpu_resource_types = {
        'c77': 'Epyc Rome',
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

    from scipy.stats import *

    for resourceid in resource_data.keys():
        for version in resource_data[resourceid].keys():
            if version != '0':
                continue
            csv_filename = resourceid + version + '.csv'
            data = {version: resource_data[resourceid][version]}
            #print (resourceid, version)

            shape, location, scale = gamma.fit(resource_data[resourceid][version])
            # mu, sigma = np.log(scale), shape
            print(resourceid, version, shape, location, scale)

            ks0 = stats.kstest(resource_data[resourceid][version], 'gamma', args=[shape, location, scale])
            #print(ks0)

            x = np.linspace(min(resource_data[resourceid][version]), max(resource_data[resourceid][version]), 500)
            y = gamma.pdf(x, shape, location, scale)

            '''
            min_x = None
            max_x = None
            avg_x = 0
            samples = []
            for i in range(0, 10000):
                sample = np.random.gamma(shape, scale) + min(resource_data[resourceid][version])
                samples.append (sample)
                if min_x == None:
                    min_x = sample
                else:
                    if min_x > sample:
                        min_x = sample
                if max_x == None:
                    max_x = sample
                else:
                    if max_x < sample:
                        max_x = sample

                avg_x += sample
            print(min_x, max_x, avg_x / 10000, np.median(samples), mode (samples))
            '''
            samples1 = gamma.rvs(shape, location, scale, size=10000)

            print(min(samples1), max(samples1), statistics.mean(samples1), np.median(samples1), mode(samples1))

            #plt.plot(x, y)
            #plt.hist(resource_data[resourceid][version], bins=30)
            #plt.show()
            print ("###########")
            '''
            #ks1 = stats.kstest(resource_data[resourceid][version], 'gamma', args=[mu, 0, sigma])
            #dataframe = pd.DataFrame.from_dict(data)
            #dataframe.to_csv(csv_filename, index=False)
            '''

    exit (0)

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
            if version != '3':
                continue
            csv_filename = resourceid + version + '.csv'
            data = {version: resource_data[resourceid][version]}


            shape, location, scale = gamma.fit(resource_data[resourceid][version])
            # mu, sigma = np.log(scale), shape
            print(resourceid, version, shape, location, scale)

            ks0 = stats.kstest(resource_data[resourceid][version], 'gamma', args=[shape, location, scale])
            print(ks0)

            x = np.linspace(min(resource_data[resourceid][version]), max(resource_data[resourceid][version]), 500)
            y = gamma.pdf(x, shape, location, scale)

            '''
            min_x = None
            max_x = None
            avg_x = 0
            samples = []
            for i in range(0, 10000):
                sample = np.random.lognorm(shape, scale) + min(resource_data[resourceid][version])
                samples.append (sample)
                if min_x == None:
                    min_x = sample
                else:
                    if min_x > sample:
                        min_x = sample
                if max_x == None:
                    max_x = sample
                else:
                    if max_x < sample:
                        max_x = sample

                avg_x += sample
            print(min_x, max_x, avg_x / 10000, np.median(samples), mode (samples))
            '''
            samples1 = gamma.rvs(shape, location, scale, size=10000)

            print (min (samples1), max (samples1), statistics.mean(samples1), np.median(samples1), mode (samples1))

            #plt.plot(x, y)
            #plt.hist(resource_data[resourceid][version], bins=30)
            #plt.show()

            '''
            print (resourceid, version)
            f = Fitter(resource_data[resourceid][version])
            f.fit()
            print(f.summary())
            print (f.get_best(method = 'sumsquare_error'))
            '''
            print("###########")
            #dataframe = pd.DataFrame.from_dict(data)
            #dataframe.to_csv(csv_filename, index=False)


    '''
    from scipy.stats import *

    distributions = ['alpha', 'anglit', 'arcsine', 'argus', 'beta', 'betaprime', 'bradford', 'burr', 'burr12', 'cauchy', 'chi','chi2', 'cosine', 'crystalball',
                     'dgamma', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'genlogistic',
                     'gennorm', 'genpareto', 'genexpon', 'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'genhyperbolic', 'geninvgauss', 'gilbrat', 'gompertz',
                     'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'halfgennorm', 'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'kappa4',
                     'kappa3', 'ksone', 'kstwo', 'kstwobign', 'laplace', 'laplace_asymmetric', 'levy', 'levy_l', 'levy_stable', 'logistic', 'loggamma', 'loglaplace', 'lognorm', 'loguniform', 'lomax',
                     'maxwell', 'mielke', 'moyal', 'akagami', 'ncx2', 'ncf', 'nct', 'norm', 'norminvgauss', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'rayleigh', 'rice', 'recipinvgauss',
                     'semicircular', 'skewcauchy', 'skewnorm', 'studentized_range', 't', 'trapezoid', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'vonmises_line', 'wald', 'weibull_min', 'weibull_max', 'wrapcauchy']

    distributions = ["lognorm"]

    #x = kstest(resource_data['c77']['0'], "anglit")

    resourceids = ['c99']

    for resourceid in resourceids:
        for version in resource_data[resourceid].keys():
            print("###################")
            shape, location, scale = lognorm.fit(resource_data[resourceid][version])
            mu, sigma = np.log(scale), shape
            print (resourceid, version, shape, location, scale, mu, sigma)
            ks0 = stats.kstest(resource_data[resourceid][version], 'lognorm', args=[shape, location, scale])
            ks1 = stats.kstest(resource_data[resourceid][version], 'lognorm', args=[mu, 0, sigma])
            print (ks0, ks1)
            print ("###################")
            plt.hist(resource_data[resourceid][version], bins=30)
            plt.show()

    for distribution in distributions:
        print (resource_data['c77']['0'], distribution)
        x = kstest(resource_data['c77']['0'], distribution)
        print (x)
    for resourcetype in resource_data.keys():
        for version in resource_data[resourcetype].keys():
            print (resourcetype, version)

    #plt.show ()

    #plt.savefig('greedyoptimiation.png', dpi=400)
    '''