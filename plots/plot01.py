import matplotlib.pyplot as plt
import json, sys
import pandas as pd
import datetime
import numpy as np

avg_time = []

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

resource_performance = {}

def plot_results (results_file):
    global avg_time

    file_handle = open(results_file, 'r')
    lines = file_handle.readlines()
    earliest_start_time = None
    last_end_time = None

    index = 0
    my_count = 0
    for line in lines:
        line = line.strip()
        if 'probe_status (complete):' in line:

            print(line)
            _, _, image_id, version, starttime_date, starttime_time, endtime_date, endtime_time, resource_id = line.split(' ', 8)

            starttime = starttime_date + ' ' + starttime_time
            endtime = endtime_date + ' ' + endtime_time
            starttime = datetime.datetime.strptime(starttime,
                                                   '%Y-%m-%d %H:%M:%S')
            endtime = datetime.datetime.strptime(endtime, '%Y-%m-%d %H:%M:%S')

            if earliest_start_time == None:
                earliest_start_time = starttime
            else:
                if earliest_start_time > starttime:
                    earliest_start_time = starttime
            if last_end_time == None:
                last_end_time = endtime
            else:
                if last_end_time < endtime:
                    last_end_time = endtime

            if resource_id in resource_performance.keys():
                resource_performance_queue = resource_performance[resource_id]
                resource_performance_queue.append([image_id, version, starttime, endtime])
            else:
                resource_performance_queue = []
                resource_performance_queue.append([image_id, version, starttime, endtime])
                resource_performance[resource_id] = resource_performance_queue
        index += 1
    print (my_count)

    print(resource_performance)
    print ((last_end_time - earliest_start_time).total_seconds())


    zero_df = pd.DataFrame()
    one_df = pd.DataFrame()
    two_df = pd.DataFrame()
    three_df = pd.DataFrame()
    four_df = pd.DataFrame()

    zero_df['resources'] = pd.Series(list(resource_performance.keys()))
    one_df['resources'] = pd.Series(list(resource_performance.keys()))
    two_df['resources'] = pd.Series(list(resource_performance.keys()))
    three_df['resources'] = pd.Series(list(resource_performance.keys()))
    four_df['resources'] = pd.Series(list(resource_performance.keys()))

    performance_columns = ['resources', '0', '1', '2', '3', '4']


    zero_count = []
    one_count = []
    two_count = []
    three_count = []
    four_count = []

    zero_time = []
    one_time = []
    two_time = []
    three_time = []
    four_time = []

    for resource_id in resource_performance.keys():
        zero = one = two = three = four = 0
        avg_time_0 = 0
        avg_time_1 = 0
        avg_time_2 = 0
        avg_time_3 = 0
        avg_time_4 = 0
        for image in resource_performance[resource_id]:

            id = int (image[1]) % 5
            if id == 0:
                zero += 1
                avg_time_0 += (image[3] - image[2]).total_seconds()
            elif id == 1:
                one += 1
                avg_time_1 += (image[3] - image[2]).total_seconds()
            elif id == 2:
                two += 1
                avg_time_2 += (image[3] - image[2]).total_seconds()
            elif id == 3:
                three += 1
                avg_time_3 += (image[3] - image[2]).total_seconds()
            elif id == 4:
                four += 1
                avg_time_4 += (image[3] - image[2]).total_seconds()

        print (resource_id, zero, one, two, three, four)

        zero_count.append(zero)
        one_count.append(one)
        two_count.append(two)
        three_count.append(three)
        four_count.append(four)

        zero_time.append(avg_time_0/zero)
        one_time.append(avg_time_1 /one)
        two_time.append(avg_time_2 / two)
        three_time.append(avg_time_3 / three)
        four_time.append(avg_time_4 / four)

    zero_df['count'] = pd.Series(zero_count)
    one_df['count'] = pd.Series(one_count)
    two_df['count'] = pd.Series(two_count)
    three_df['count'] = pd.Series(three_count)
    four_df['count'] = pd.Series(four_count)

    zero_df.set_index('resources', inplace=True)
    one_df.set_index('resources', inplace=True)
    two_df.set_index('resources', inplace=True)
    three_df.set_index('resources', inplace=True)
    four_df.set_index('resources', inplace=True)

    print (zero_df)

    avg_time.append(zero_time)
    avg_time.append(one_time)
    avg_time.append(two_time)
    avg_time.append(three_time)
    avg_time.append(four_time)

    plot_clustered_stacked([zero_df, one_df, two_df, three_df, four_df],
        ["P1", "P2", "P3", "P4", "P5"])

    plt.xlabel('Resources', fontsize=15, fontname='Times New Roman', fontweight='normal')
    plt.ylabel('Work Items Count', fontsize=15, fontname='Times New Roman', fontweight='normal')
    plt.yticks(fontsize=15, fontname='Times New Roman', fontweight='normal')

    plt.show()

if __name__ == "__main__":
    results_file = "run_0_50_51_96_98_44_47_First_random/output.txt"
    plot_results(results_file)