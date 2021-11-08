import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import datetime
import statistics
from matplotlib.lines import Line2D

matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['font.family'] = 'Times New Roman'

x_axis = ['c44', 'c77', 'c93', 'c98', 'c99']

data1 = [
    [22.3, 14.9, 15.9, 13.1, 12.3],
    [389.9, 196.1, 373.9, 192.8, 267.7],
    [272.9, 154.2, 295.7, 158.4, 204.9]
]

data2 = [
    [22, 14.1, 15.7, 12.4, 12.3],
    [385.9, 211.1, 370.1, 193.1, 266.8],
    [278.4, 170.2, 299.5, 163.8, 209.4]
]

if __name__ == "__main__":
    #global data1, data2

    labels = ['stage1', 'stage3', 'stage5']

    fig, axes = plt.subplots(nrows=3, sharex=True)

    colors = ['r', 'b']

    index = 0
    for data_1, data_2 in zip(data1, data2):
        axes[index].plot(x_axis, data_1, lw=4, color= colors[0])
        axes[index].plot(x_axis, data_2, lw=4, color= colors[1])

        axes[index].yaxis.set_label_position("right")

        axes[index].set_ylabel(labels[index])
        index += 1

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)

    plt.xlabel("Nodes")
    plt.ylabel("avg. execution time (seconds)")

    lines = []

    for color in colors:
        lines.append(Line2D([0], [0], color=color, lw=2, linestyle='solid'))

    legends = ['CPU, GPU co-located', 'CPU, GPU separated']

    plt.legend (lines, legends)

    plt.savefig('cpugpucolocationeffect.png', dpi=400)

    #completion_file1 = open("run_39_10_44_First_0/complete.txt", "r")
    #completion_file2 = open("run_48_10_44_93_First_0/complete.txt", "r")

    #completion_file1 = open("run_40_10_93_First_0/complete.txt", "r")
    #completion_file2 = open("run_52_10_93_98_First_0/complete.txt", "r")

    #completion_file1 = open("run_41_10_98_First_0/complete.txt", "r")
    #completion_file2 = open("run_49_10_98_93_First_0/complete.txt", "r")

    #completion_file1 = open("run_42_10_77_First_0/complete.txt", "r")
    #completion_file2 = open("run_51_10_77_93_First_0/complete.txt", "r")

    completion_file1 = open("run_43_10_99_First_0/complete.txt", "r")
    completion_file2 = open("run_50_10_99_93_First_0/complete.txt", "r")

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