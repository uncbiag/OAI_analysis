import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import datetime
import statistics
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D

matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['font.family'] = 'Times New Roman'

if __name__ == "__main__":
    completion_file1 = open("run_25_50_44_51_77_93_98_Fast_0_desc/complete.txt", "r")
    #
    completion_file2 = open("run_21_50_44_51_77_93_98_First_0/complete.txt", "r")
    completion_lines1 = completion_file1.readlines()
    completion_lines2 = completion_file2.readlines()

    versions1 = {}

    versions1['0'] = {}
    versions1['1'] = {}
    versions1['2'] = {}
    versions1['3'] = {}
    versions1['4'] = {}

    versions2 = {}

    versions2['0'] = {}
    versions2['1'] = {}
    versions2['2'] = {}
    versions2['3'] = {}
    versions2['4'] = {}

    for completion_line1 in completion_lines1:
        _, _, imageid, version, _, _, startdate, starthour, enddate, endhour, resourceid, _ = completion_line1.split(' ', 11)
        resourceid = resourceid.strip ()
        #_, _, imageid, version, startdate, starthour, enddate, endhour, resourceid = completion_line.split(' ', 8)
        starttime_s = startdate + ' ' + starthour
        endtime_s = enddate + ' ' + endhour
        starttime = datetime.datetime.strptime(starttime_s,
                                               '%Y-%m-%d %H:%M:%S').timestamp()
        endtime = datetime.datetime.strptime(endtime_s, '%Y-%m-%d %H:%M:%S').timestamp()

        version_data = versions1[version]

        if resourceid in version_data.keys():
            version_data[resourceid].append (endtime - starttime)
        else:
            version_data[resourceid] = []
            version_data[resourceid].append (endtime - starttime)

    for completion_line2 in completion_lines2:
        _, _, imageid, version, _, _, startdate, starthour, enddate, endhour, resourceid, _ = completion_line2.split(' ', 11)
        resourceid = resourceid.strip ()
        #_, _, imageid, version, startdate, starthour, enddate, endhour, resourceid = completion_line.split(' ', 8)
        starttime_s = startdate + ' ' + starthour
        endtime_s = enddate + ' ' + endhour
        starttime = datetime.datetime.strptime(starttime_s,
                                               '%Y-%m-%d %H:%M:%S').timestamp()
        endtime = datetime.datetime.strptime(endtime_s, '%Y-%m-%d %H:%M:%S').timestamp()

        version_data = versions2[version]

        if resourceid in version_data.keys():
            version_data[resourceid].append (endtime - starttime)
        else:
            version_data[resourceid] = []
            version_data[resourceid].append (endtime - starttime)


    fig, axes = plt.subplots(5, 1, sharex=True)

    labels = ['stage1', 'stage2', 'stage3', 'stage3', 'stage4', 'stage5']

    index = 0
    for version_key in versions1:
        ax = axes[index]
        version_data1 = versions1[version_key]
        version_data2 = versions2[version_key]

        ax_min1 = []
        ax_max1 = []
        ax_mean1 = []
        ax_stddev1 = []
        counts1 = []

        ax_min2 = []
        ax_max2 = []
        ax_mean2 = []
        ax_stddev2 = []
        counts2 = []

        resource_keys = ['c44', 'c51', 'c93', 'c98', 'c77']

        trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
        trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData

        print (version_data1)
        print(version_data2)

        for resource in resource_keys:
            resource_data1 = version_data1[resource]
            print (resource, resource_data1)
            #print (resource_data, min(resource_data), max(resource_data), sum(resource_data)/len(resource_data), statistics.stdev(resource_data))
            ax_min1.append (min(resource_data1))
            ax_max1.append (max(resource_data1))
            ax_mean1.append(sum(resource_data1)/len(resource_data1))
            if len (resource_data1) == 1:
                ax_stddev1.append(0)
            else:
                ax_stddev1.append(statistics.stdev(resource_data1))
            counts1.append (len (resource_data1))

            resource_data2 = version_data2[resource]
            # print (resource_data, min(resource_data), max(resource_data), sum(resource_data)/len(resource_data), statistics.stdev(resource_data))
            ax_min2.append(min(resource_data2))
            ax_max2.append(max(resource_data2))
            ax_mean2.append(sum(resource_data2) / len(resource_data2))
            if len (resource_data2) == 1:
                ax_stddev2.append(0)
            else:
                ax_stddev2.append(statistics.stdev(resource_data2))
            counts2.append(len(resource_data2))

        ax_min1 = np.array(ax_min1)
        ax_mean1 = np.array(ax_mean1)
        ax_max1 = np.array(ax_max1)
        ax_stddev1 = np.array(ax_stddev1)

        ax_min2 = np.array(ax_min2)
        ax_mean2 = np.array(ax_mean2)
        ax_max2 = np.array(ax_max2)
        ax_stddev2 = np.array(ax_stddev2)

        ax.errorbar(resource_keys, ax_mean1, ax_stddev1, fmt='ok', lw=3, transform = trans1)
        ax.errorbar(resource_keys, ax_mean1, [ax_mean1 - ax_min1, ax_max1 - ax_mean1],
                     fmt='.k', ecolor='gray', lw=1, transform = trans1)


        ax.plot (resource_keys, ax_mean1, lw=2, transform = trans1, color='r')

        xdata1 = [0.1, 1.1, 2.1, 3.1, 4.1]
        xdata2 = [-0.1, .9, 1.9, 2.9, 3.9]

        for i, txt in enumerate(counts1):
            ax.annotate(txt, (xdata2[i], ax_mean1[i]), transform='data')

        ax.errorbar(resource_keys, ax_mean2, ax_stddev2, fmt='ok', lw=3, transform=trans2)
        ax.errorbar(resource_keys, ax_mean2, [ax_mean2 - ax_min2, ax_max2 - ax_mean2],
                    fmt='.k', ecolor='gray', lw=1, transform=trans2)

        ax.plot(resource_keys, ax_mean2, lw=2, transform=trans2, color='g')

        for i, txt in enumerate(counts2):
            ax.annotate(txt, (xdata1[i], ax_mean2[i]), transform='data')

        lines = []

        lines.append(Line2D([0], [0], color='r', lw=2, linestyle='solid'))
        lines.append(Line2D([0], [0], color='g', lw=2, linestyle='solid'))

        legends =['FAST', 'FIRST']

        ax.legend(lines, legends, prop={'size': 6})

        ax.set_ylabel (labels[index])

        index += 1

    plt.xlabel('resources')

    plt.show()

    #plt.savefig('stagecomparison.png', dpi=400)


