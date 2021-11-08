import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import datetime
import statistics
from matplotlib.lines import Line2D

matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['font.family'] = 'Times New Roman'

if __name__ == "__main__":
    completion_file = open("run_33_100_44_51_77_93_98_First_0/complete.txt", "r")
    #completion_file = open("run_13_50_44_47_75_78_98_First_earliest_sched/complete.txt", "r")
    completion_lines = completion_file.readlines()

    resource_data = {}

    min_start_time = None
    max_end_time = None

    images = []

    for completion_line in completion_lines:
        _, _, imageid, version, _, _, startdate, starthour, enddate, endhour, resourceid = completion_line.split(' ',
                                                                                                                 10)
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


        if max_end_time == None:
            max_end_time = endtime
        elif max_end_time < endtime:
            max_end_time = endtime

        images.append([imageid, version, resourceid, starttime, endtime])

    for image in images:
        image[3] -= min_start_time
        image[4] -= min_start_time
        print (image)

    for image in images:
        imageid = image[0]
        version = image[1]
        resourceid = image[2]
        starttime = image[3]
        endtime = image[4]

        if resourceid not in resource_data:
            resource_data[resourceid] = {}
            if int(version) % 2 == 0:
                resource_data[resourceid]['CPU'] = []
                resource_data[resourceid]['CPU'].append ([imageid, starttime, endtime, version])
            else:
                resource_data[resourceid]['GPU'] = []
                resource_data[resourceid]['GPU'].append([imageid, starttime, endtime, version])

        else:
            if int(version) % 2 == 0:
                if 'CPU' not in resource_data[resourceid]:
                    resource_data[resourceid]['CPU'] = []
                    resource_data[resourceid]['CPU'].append ([imageid, starttime, endtime, version])
                else:
                    resource_data[resourceid]['CPU'].append ([imageid, starttime, endtime, version])
            else:
                if 'GPU' not in resource_data[resourceid]:
                    resource_data[resourceid]['GPU'] = []
                    resource_data[resourceid]['GPU'].append([imageid, starttime, endtime, version])
                else:
                    resource_data[resourceid]['GPU'].append([imageid, starttime, endtime, version])

    print (resource_data)

    y = []
    x = []
    styles = []
    colors = []
    index = 0

    colors_data = ['red', 'black', 'green', 'pink', 'blue']

    for resourceid in resource_data:
        for cpu_data in resource_data[resourceid]['CPU']:
            x.append ([cpu_data[1], cpu_data[2]])
            y.append ([index, index])
            styles.append ('solid')
            colors.append (colors_data[int(cpu_data[3])])
        index += 1
        for gpu_data in resource_data[resourceid]['GPU']:
            x.append ([gpu_data[1], gpu_data[2]])
            y.append ([index, index])
            styles.append ('dotted')
            colors.append (colors_data[int(gpu_data[3])])
        index += 1

    fig, axe = plt.subplots()
    for xaxis, yaxis, color, style in zip(x, y, colors, styles):
        plt.plot(xaxis, yaxis, lw=4, color=color, linestyle=style)

    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['C44(C)', 'C44(G)', 'C47(C)', 'C47(G)', 'C75(C)', 'C75(G)', 'C78(C)', 'C78(G)', 'C98(C)', 'C98(G)'])

    plt.xlim(0, 7000)

    plt.xlabel ('timeline (seconds)')

    lines = []

    for resource_color in colors_data:
        lines.append(Line2D([0], [0], color=resource_color, lw=2, linestyle='solid'))

    legends = ['stage 0', 'stage 1', 'stage 2', 'stage 3', 'stage 4']

    axe.legend (lines, legends)

    plt.show ()

    #plt.savefig('fast-0-4.png', dpi=400)

