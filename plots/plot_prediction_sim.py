import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import datetime
import statistics

matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['font.family'] = 'Times New Roman'

#current_time, index, pipelinestage.name.split(':')[0],
#phase.starttime, phase.pstarttime, phase.pendtime,
#phase.pending_output,
#phase.pfirst_workitem_completion_time,
#phase.pfirst_resource_release_time

def plot_prediction_sim (plot_data, prediction_times, batchsize):
    fig, axes = plt.subplots(5, 1, sharex=True)

    labels = ['stage1', 'stage2', 'stage3', 'stage4', 'stage5']

    pipelinestage_index = 0
    for pipelinestage_name in plot_data:
        phases_data = plot_data[pipelinestage_name]
        phase_index = 0
        ax = axes[pipelinestage_index]

        ax.set_ylim (bottom=-2, top=batchsize)

        for phase_data in phases_data:
            x_data = []
            y_data = []
            queued_snapshots = phase_data[0]
            phase_starttime = phase_data[1]
            phase_endtime = phase_data[2]

            for key in queued_snapshots.keys ():
                time = int (float(key) * 3600)
                value = queued_snapshots[key]
                x_data.append(time)
                y_data.append(value)

            p = ax.plot(x_data, y_data)
            ax.plot ([int (float (phase_starttime) * 3600), int (float (phase_starttime) * 3600)] , [1, -1], p[0].get_color ())
            ax.plot ([int (float (phase_endtime) * 3600), int (float (phase_endtime) * 3600)], [1, -1], p[0].get_color ())
            ax.plot ([int (float (phase_starttime) * 3600), int (float (phase_endtime) * 3600)] , [0, 0], p[0].get_color ())

            real_timediff = int (float (phase_endtime) * 3600) - int (float (phase_starttime) * 3600)

            ax.text ((int (float (phase_starttime) * 3600) + int (float (phase_endtime) * 3600))/2, 0, str(real_timediff), fontsize=10)

            y_limit = ax.get_ylim()[1] - 1

            prediction_index = 0
            phase_predictions = phase_data[3]
            for prediction_time in phase_predictions:
                prediction = phase_predictions[prediction_time]
                prediction_starttime = prediction[4]
                prediction_endtime = prediction[5]

                #ax.plot([int(float(prediction_starttime) * 3600), int(float(prediction_starttime) * 3600)], [1, -1], p[0].get_color())
                #ax.plot([int(float(prediction_endtime) * 3600), int(float(prediction_endtime) * 3600)], [1, -1], p[0].get_color())

                ax.plot([int(float(prediction_starttime) * 3600), int(float(prediction_starttime) * 3600)], [y_limit - prediction_index, y_limit - (prediction_index + 2)], p[0].get_color())
                ax.plot([int(float(prediction_endtime) * 3600), int(float(prediction_endtime) * 3600)], [y_limit - prediction_index, y_limit - (prediction_index + 2)], p[0].get_color())

                ax.plot([int(float(prediction_starttime) * 3600), int(float(prediction_endtime) * 3600)],
                        [y_limit - (prediction_index + 1), y_limit - (prediction_index + 1)], p[0].get_color())

                prediction_timediff = int(float(prediction_endtime) * 3600) - int(float(prediction_starttime) * 3600)

                percentage_diff = round ((real_timediff - prediction_timediff) / prediction_timediff * 100, 2)

                ax.text (int(float(prediction_endtime) * 3600), y_limit - (prediction_index + 1), str (prediction_timediff) + '[' + str(percentage_diff) + '%]', fontsize=10)

                prediction_index += 2

            phase_index += 1

        ax.yaxis.set_label_position("right")
        ax.set_ylabel(labels[pipelinestage_index])
        pipelinestage_index += 1

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)

    plt.xlabel("Timeline (seconds)")
    plt.ylabel("Queue size")

    # plt.xlabel('Timeline (seconds)')

    plt.show()
    #plt.savefig('tmp.png', dpi=300)