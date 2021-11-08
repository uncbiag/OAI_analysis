import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import datetime
import statistics
from matplotlib.lines import Line2D

matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['font.family'] = 'Times New Roman'

x_axis = ['c44', 'c77', 'c93', 'c98', 'c99']

image_sizes = [250, 200, 150, 100, 50]

data1 = [23177, 19252, 14431, 9914, 5023]
data2 = [23558, 18816, 14400, 9730, 4760]

if __name__ == "__main__":
    plt.plot (image_sizes, data1, color = 'r')
    plt.plot (image_sizes, data2, color = 'g')

    plt.xlabel("No. of images")
    plt.ylabel("completion time (seconds)")

    plt.ylim(0, 30000)

    plt.savefig('simcomparison.png', dpi=400)

    #plt.savefig('cpugpucolocationeffect.png', dpi=400)


    #completion_file1 = open("run_39_10_44_First_0/complete.txt", "r")
    #completion_file2 = open("run_48_10_44_93_First_0/complete.txt", "r")

    #completion_file1 = open("run_40_10_93_First_0/complete.txt", "r")
    #completion_file2 = open("run_52_10_93_98_First_0/complete.txt", "r")

    #completion_file1 = open("run_41_10_98_First_0/complete.txt", "r")
    #completion_file2 = open("run_49_10_98_93_First_0/complete.txt", "r")

    #completion_file1 = open("run_42_10_77_First_0/complete.txt", "r")
    #completion_file2 = open("run_51_10_77_93_First_0/complete.txt", "r")

    #completion_file1 = open("run_43_10_99_First_0/complete.txt", "r")
    #completion_file2 = open("run_50_10_99_93_First_0/complete.txt", "r")

    completion_files = [completion_file1, completion_file2]

    resource_datas = {}

    print (completion_files)

    for completion_file in completion_files:
        completion_lines = completion_file.readlines()

        version_data = {}

        min_start_time = None
        max_end_time = None

        images = []

        for completion_line in completion_lines:
            _, _, imageid, version, _, _, startdate, starthour, enddate, endhour, resourceid, _ = completion_line.split(
                ' ', 11)

            resourceid = resourceid.strip()

            starttime_s = startdate + ' ' + starthour
            endtime_s = enddate + ' ' + endhour
            starttime = datetime.datetime.strptime(starttime_s,
                                                   '%Y-%m-%d %H:%M:%S').timestamp()
            endtime = datetime.datetime.strptime(endtime_s, '%Y-%m-%d %H:%M:%S').timestamp()


            if min_start_time == None:
                min_start_time = starttime
            elif min_start_time > starttime:
                min_start_time = starttime

            images.append([imageid, version, resourceid, starttime, endtime])


        for image in images:
            image[3] -= min_start_time
            image[4] -= min_start_time

        for image in images:
            imageid = image[0]
            version = image[1]
            resourceid = image[2]
            starttime = image[3]
            endtime = image[4]

            if version not in version_data:
                if int(version) % 2 == 0:
                    version_data[version] = []
                    version_data[version].append (endtime - starttime)
            else:
                if int(version) % 2 == 0:
                    version_data[version].append(endtime - starttime)

        for version in version_data:
            datapoints = version_data[version]
            print (version, sum(datapoints)/len(datapoints))