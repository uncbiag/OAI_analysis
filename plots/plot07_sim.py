import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import datetime
import statistics

matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['font.family'] = 'Times New Roman'

if __name__ == "__main__":
    imagesizes = [50, 100, 150, 200, 250]
    output_directory = "DFS_staging"

    total_times = []

    for i in range(len(imagesizes)):
        completion_file = open (output_directory+ "/" + str(imagesizes[i]) + ".txt", "r")

        completion_lines = completion_file.readlines()

        print(len(completion_lines))

        resource_data = {}

        min_starttime = None

        images = []

        for completion_line in completion_lines:
            _, _, imageid, version, starttime, endtime, resourceid, _ = completion_line.split(' ', 7)

            resourceid = resourceid.strip()


            if min_starttime == None:
                min_starttime = starttime
            elif min_starttime > starttime:
                min_starttime = starttime

            images.append([imageid, version, resourceid, starttime, endtime])

        #images.sort(key=lambda x: x[4])

        total_time = images[-1][4]

        print (float(total_time) * 3600)

        total_times.append(total_time)


    #plt.plot(imagesizes, total_times, lw=4)

    #plt.xticks(imagesizes)

    #plt.xlabel("No. of Images")
    #plt.ylabel("Time Taken (hrs)")

    #plt.xlabel('Timeline (seconds)')

    #plt.show()
    #plt.savefig('queued.png', dpi=400)
