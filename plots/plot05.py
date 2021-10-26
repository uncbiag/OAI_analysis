import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import datetime
import statistics

matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['font.family'] = 'Times New Roman'

def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H="//", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    print (h, l)
    colors = ['r', 'b', 'y', 'c', 'g']
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            app_index = 0
            print (len(pa.patches))
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col) - 0.1)
                rect.set_width(1 / float(n_df + 1))
                rect.set_color (colors[i])
                axe.text(rect.get_x(), rect.get_width() * 3, str(round(avg_time[i][app_index],2)) + ' seconds', fontsize=15,
                         fontname='Times New Roman', rotation=70, fontweight='normal')
                app_index += 1

    #axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0, fontsize=15, fontname='Times New Roman', fontweight='normal')
    #axe.set_yticks(fontsize=10, fontname ='Times New Roman')
    #axe.set_yticklabels([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], rotation = 0, fontsize=15, fontname='Times New Roman', fontweight='normal')
    axe.set_ylim(0, 25)

    # Add invisible data to add another legend
    n=[]
    for i in range(n_df):
        if i == 1:
            H = '\\\\'
        elif i == 2:
            H = '/'
        elif i == 3:
            H = '\\'
        elif i == 4:
            H = '/'
        n.append(axe.bar(0, 0, color=colors[i], width=2))

    legend_properties = {'size': '15', 'family':'Times New Roman', 'weight':'normal'}
    bars = h[:n_col]
    bars.reverse()
    #l1 = axe.legend(bars, l[:n_col], prop=legend_properties, loc=1, borderaxespad=0, borderpad = 0.2)

    if labels is not None:
        l2 = plt.legend(n, labels, prop=legend_properties, loc=2, borderaxespad=0, borderpad = 0.2, ncol=2)
    #axe.add_artist(l1)
    return axe

if __name__ == "__main__":
    #completion_file = open("run_12_50_44_47_75_78_98_Fast/complete.txt", "r")
    completion_file = open("run_13_50_44_47_75_78_98_First_earliest_sched/complete.txt", "r")
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

    unique_images_data = {}
    for image in images:
        if image[0] in unique_images_data:
            unique_images_data[image[0]].append([image[3], image[4], image[2], image[1]])
        else:
            unique_images_data[image[0]] = []
            unique_images_data[image[0]].append([image[3], image[4], image[2], image[1]])

    unique_images_data = dict(sorted(unique_images_data.items(), key=lambda item: item[1][0][0]))

    print (unique_images_data)

    image_waiting_data = {}

    for image in unique_images_data:
        index = 0
        for version_data in unique_images_data[image]:
            if version_data[3] == '0':
                index += 1
                continue
            else:
                if image not in image_waiting_data:
                    image_waiting_data[image] = []
                    image_waiting_data[image].append (version_data[0] - unique_images_data[image][index - 1][1])
                else:
                    image_waiting_data[image].append(
                        version_data[0] - unique_images_data[image][index - 1][1])
                index += 1

    print (image_waiting_data)

    y = []
    x = []
    colors = []
    styles = []

    colors_data = ['red', 'black', 'green', 'pink']

    index = 0
    for image_id in image_waiting_data:
        waiting_times = image_waiting_data[image_id]

        count = 0
        sum = 0
        for waiting_time in waiting_times:
            y.append([index, index])
            if count == 0:
                x.append ([0, waiting_time])
                sum += waiting_time
            else:
                x.append ([sum, sum + waiting_times[count]])
                sum += waiting_time
            colors.append(colors_data[count])

            styles.append('solid')
            count += 1
        index += 1

    print (y)
    print (x)

    fig, axe = plt.subplots()

    for xaxis, yaxis, color, style in zip(x, y, colors, styles):
        plt.plot(xaxis, yaxis, lw=4, color=color, linestyle=style)

    plt.show()
