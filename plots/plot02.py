import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import datetime
from matplotlib.lines import Line2D

matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['font.family'] = 'Times New Roman'

if __name__ == '__main__':
    #completion_file = open("run_12_50_44_47_75_78_98_Fast/complete.txt", "r")
    #completion_file = open("run_13_50_44_47_75_78_98_First_earliest_sched/complete.txt", "r")
    #completion_file = open("run_14_50_44_47_75_78_98_First_earliest_finish/complete.txt", "r")
    #completion_file = open("run_15_50_44_47_75_78_98_First_latest_finish/complete.txt", "r")
    #completion_file = open("run_16_50_44_51_77_93_94_Fast_0/complete.txt", "r")
    #completion_file = open("run_17_50_44_51_77_93_94_First_0/complete.txt", "r")
    #completion_file = open("run_18_50_44_51_77_93_98_First_0/complete.txt", "r")
    #completion_file = open("run_19_50_44_51_77_93_98_Fast_0/complete.txt", "r")
    #completion_file = open("run_21_50_44_51_77_93_98_First_0/complete.txt", "r") #5262
    #completion_file = open("run_22_50_44_51_77_93_98_Fast_0_desc_newwm/complete.txt", "r") #5358
    #completion_file = open("run_23_50_44_51_77_93_98_DFS/complete.txt", "r")#6277
    #completion_file = open("run_24_50_44_51_77_93_98_Fast_1_desc_newwm/complete.txt", "r") #5257
    #completion_file = open("run_25_50_44_51_77_93_98_Fast_0_desc/complete.txt", "r") #5263
    #completion_file = open("run_26_50_44_51_77_93_98_Fast_1_desc/complete.txt", "r") #5306
    #completion_file = open("run_27_50_44_51_77_93_98_Fast_0/complete.txt", "r") #5456
    #completion_file = open("run_28_50_44_51_77_93_98_Fast_1/complete.txt", "r") #5252
    #completion_file = open("run_29_50_44_51_77_93_98_Fast_0_desc_alloc/complete.txt", "r")#5237
    #completion_file = open("run_30_50_44_51_77_93_98_Fast_1_desc_alloc/complete.txt", "r")#5338
    # completion_file = open("run_31_50_44_51_77_93_98_Fast_0_desc_newwm_alloc/complete.txt", "r")#5494
    # completion_file = open("run_32_50_44_51_77_93_98_Fast_0_desc_newwm_alloc_final/complete.txt", "r")#5393
    #completion_file = open("run_33_100_44_51_77_93_98_First_0/complete.txt", "r")#10480
    #completion_file = open("run_34_100_44_51_77_93_98_DFS/complete.txt", "r")#12753
    completion_file = open("run_35_100_44_51_77_93_98_Fast_0_desc_alloc/complete.txt", "r")#10299
    #completion_file = open("run_36_50_44_51_77_93_98_Fast_1_desc_alloc/complete.txt", "r")
    #completion_file = open("run_37_100_44_47_77_93_98_Fast_0_desc_alloc/complete.txt", "r")#10273
    #completion_file = open("run_38_100_44_47_77_93_98_First_0/complete.txt", "r")#10316
    completion_lines = completion_file.readlines()

    images = []
    unique_images = []

    min_start_time = None
    max_end_time = None

    for completion_line in completion_lines:
        _, _, imageid, version, _, _, startdate, starthour, enddate, endhour, resourceid, _ = completion_line.split(' ', 11)
        #_, _, imageid, version, startdate, starthour, enddate, endhour, resourceid = completion_line.split(' ', 8)
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

        resourceid = resourceid.strip()
        images.append ([imageid, version, resourceid, starttime, endtime])

        if imageid not in unique_images:
            unique_images.append(imageid)
    print (min_start_time, max_end_time)

    print ('total_seconds:', max_end_time - min_start_time)

    for image in images:
        image[3] -= min_start_time
        image[4] -= min_start_time

    unique_images_data = {}
    for image in images:
        if image[0] in unique_images_data:
            unique_images_data[image[0]].append([image[3], image[4], image[2]])
        else:
            unique_images_data[image[0]] = []
            unique_images_data[image[0]].append([image[3], image[4], image[2]])

    unique_images_data = dict(sorted(unique_images_data.items(), key=lambda item: item[1][0][0]))

    resource_colors = {'c44':'red', 'c51':'blue', 'c93':'pink', 'c98':'green', 'c77': 'black'}
    y = []
    x = []
    colors = []
    styles = []
    index = 0
    for unique_image_id in unique_images_data:
        unique_images = unique_images_data[unique_image_id]
        print (index, len(unique_images))
        version = 0
        for unique_image in unique_images:
            print (index, unique_image_id, unique_image, unique_image[1] - unique_image[0])
            y.append([index, index])
            x.append([unique_image[0], unique_image[1]])
            colors.append(resource_colors[unique_image[2]])
            if version % 2 == 0:
                styles.append('solid')
            else:
                styles.append('dotted')
            version += 1
        index += 1

    print (colors)

    fig,axe = plt.subplots()

    for xaxis, yaxis, color, style in zip (x, y, colors, styles):
        plt.plot(xaxis, yaxis, lw=4, color=color, linestyle=style)

    lines = []

    for resource_color in resource_colors:
        lines.append(Line2D([0], [0], color=resource_colors[resource_color], lw=2, linestyle='solid'))
        lines.append(Line2D([0], [0], color=resource_colors[resource_color], lw=2, linestyle='dotted'))

    legends = ['C44(C)', 'C44(G)', 'C51(C)', 'C51(G)', 'C93(C)', 'C93(G)', 'C98(C)', 'C98(G)', 'C77(C)', 'C77(G)']

    axe.legend (lines, legends)

    plt.xlabel('time line (seconds)')
    plt.ylabel('images')
    plt.xlim(0, 13000)

    plt.show ()

    #plt.savefig('greedyoptimiation.png', dpi=400)