import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import datetime
import statistics

matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['font.family'] = 'Times New Roman'

if __name__ == "__main__":
    completion_file = open("DFS_staging/200.txt", "r")
    completion_lines = completion_file.readlines()

    resource_data = {}

    min_submit_time = None

    images = []

    for completion_line in completion_lines:
        _, _, imageid, version, starttime, endtime, resourceid, _, _ = completion_line.split(' ', 8)

        #print (starttime, endtime)

        starttime = int (float(starttime) * float(3600))
        endtime = int (float(endtime) * float(3600))

        resourceid = resourceid.strip()

        images.append([imageid, version, resourceid, starttime, endtime])

    images.sort(key=lambda x: x[3])

    total_time = images[-1][4]


    unique_queue_data = {}

    queue_0 = [0] * (int(total_time) + 1)
    queue_1 = [0] * (int(total_time) + 1)
    queue_2 = [0] * (int(total_time) + 1)
    queue_3 = [0] * (int(total_time) + 1)
    queue_4 = [0] * (int(total_time) + 1)

    start_counts = [200, 0, 0, 0, 0]

    queues = [queue_0, queue_1, queue_2, queue_3, queue_4]

    for image in images:
        exit_queue = queues[int(image[1])]
        if int(image[1]) < len(queues) - 1:
            entry_queue = queues[int(image[1]) + 1]


        exit_queue_index = int(image[3])
        if int(image[1]) < len(queues) - 1:
            entry_queue_index = int(image[4])

        exit_queue[exit_queue_index] -= 1
        if int(image[1]) < len(queues) - 1:
            entry_queue[entry_queue_index] += 1
            print(image, str(int(image[1])), exit_queue_index, str(int(image[1]) + 1), entry_queue_index)

    for queue in queues:
        print (queue)

    data_points = []
    queue_index = 0
    for queue in queues:
        new_data_points = [[0, start_counts[queue_index]]]
        index = 0
        for count in queue:
            if count != 0:
                new_data_points.append([index, start_counts[queue_index]])
                new_data_points.append([index + 1, start_counts[queue_index] + count])
                start_counts[queue_index] += count
            index += 1
        queue_index += 1
        data_points.append(new_data_points)

    labels = ['stage1', 'stage2', 'stage3', 'stage4', 'stage5']

    #data_points = [data_points[0]]
    #labels = [labels[0]]

    for data_point in data_points:
        print(data_point)

    fig, axes = plt.subplots(5, 1, sharex=True)

    #print (fig, axes)

    index = 0
    for data_point in data_points:
        ax = axes[index]
        x = []
        y = []

        for point in data_point:
            x.append(point[0])
            y.append(point[1])
        ax.plot (x, y)

        ax.yaxis.set_label_position("right")

        ax.set_ylabel(labels[index])
        index += 1

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)

    plt.xlabel("Timeline (seconds)")
    plt.ylabel("Queue size")

    #plt.xlabel('Timeline (seconds)')

    plt.show()
    #plt.savefig('queued_sim200_equal_input.png', dpi=400)
