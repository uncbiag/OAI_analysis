import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import datetime
import statistics
import pandas as pd
import seaborn as sns

matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['font.family'] = 'Times New Roman'

#current_time, index, pipelinestage.name.split(':')[0],
#phase.starttime, phase.pstarttime, phase.pendtime,
#phase.pending_output,
#phase.pfirst_workitem_completion_time,
#phase.pfirst_resource_release_time

prediction_starttime_diff_data = {}
prediction_endtime_diff_data = {}
prediction_executiontime_diff_data = {}
pipelinestage_data = {}


def plot_prediction_sim (plot_data, prediction_times, batchsize):
    for pipelinestage_name in plot_data:
        phases_data = plot_data[pipelinestage_name]
        phase_index = 0

        for phase_data in phases_data:
            phase_starttime = int (float (phase_data[1]) * 3600)
            phase_endtime = int (float (phase_data[2]) * 3600)
            phase_predictions = phase_data[3]
            for prediction_time in phase_predictions:
                prediction = phase_predictions[prediction_time]
                prediction_starttime = int (float (prediction[4]) * 3600)
                prediction_endtime = int (float (prediction[5]) * 3600)

                prediction_starttime_diff = abs (prediction_starttime - phase_starttime)
                prediction_endtime_diff = abs (prediction_endtime - phase_endtime)

                if pipelinestage_name not in prediction_starttime_diff_data.keys ():
                    prediction_starttime_diff_data[pipelinestage_name] = []
                    prediction_starttime_diff_data[pipelinestage_name].append (prediction_starttime_diff)
                else:
                    prediction_starttime_diff_data[pipelinestage_name].append(prediction_starttime_diff)

                if pipelinestage_name not in prediction_endtime_diff_data.keys ():
                    prediction_endtime_diff_data[pipelinestage_name] = []
                    prediction_endtime_diff_data[pipelinestage_name].append (prediction_endtime_diff)
                else:
                    prediction_endtime_diff_data[pipelinestage_name].append(prediction_endtime_diff)

                prediction_diff = prediction_endtime - prediction_starttime
                actual_diff = phase_endtime - phase_starttime
                if pipelinestage_name not in prediction_executiontime_diff_data.keys ():
                    prediction_executiontime_diff_data[pipelinestage_name] = []
                    prediction_executiontime_diff_data[pipelinestage_name].append (abs (prediction_diff - actual_diff))
                else:
                    prediction_executiontime_diff_data[pipelinestage_name].append(abs (prediction_diff - actual_diff))


def plot_prediction ():
    pipelinestage_list = []
    starttime_data = []
    endtime_data = []
    executiontime_data = []

    for pipelinestage_name in prediction_starttime_diff_data.keys():
        pipelinestage_data = prediction_starttime_diff_data[pipelinestage_name]
        pipelinestage_list.extend([pipelinestage_name.split(':')[0]] * len (pipelinestage_data))
        starttime_data.extend(pipelinestage_data)

    for pipelinestage_name in prediction_endtime_diff_data.keys():
        pipelinestage_data = prediction_endtime_diff_data[pipelinestage_name]
        #pipelinestage_list.extend([pipelinestage_name.split(':')[0]] * len (pipelinestage_data))
        endtime_data.extend(pipelinestage_data)

    for pipelinestage_name in prediction_executiontime_diff_data.keys():
        pipelinestage_data = prediction_executiontime_diff_data[pipelinestage_name]
        #pipelinestage_list.extend([pipelinestage_name.split(':')[0]] * len (pipelinestage_data))
        executiontime_data.extend(pipelinestage_data)

    print (len (pipelinestage_list), len (starttime_data), len(endtime_data), len(executiontime_data))

    df = pd.DataFrame(
        {'pipelinestage':pipelinestage_list, 'starttime diff':starttime_data, 'endtime diff':endtime_data, 'executiontime diff':executiontime_data}
    )
    df = df[['pipelinestage', 'starttime diff', 'endtime diff', 'executiontime diff']]

    '''
    df_starttime = pd.DataFrame ({})
    df = pd.DataFrame({'Group': ['A', 'A', 'A', 'B', 'C', 'B', 'B', 'C', 'A', 'C'], \
                       'Apple': np.random.rand(10), 'Orange': np.random.rand(10)})
    df = df[['Group', 'Apple', 'Orange']]
    '''
    dd = pd.melt(df, id_vars=['pipelinestage'], value_vars=['starttime diff', 'endtime diff', 'executiontime diff'], var_name='Differences')
    sns.boxplot(x='pipelinestage', y='value', data=dd, hue='Differences')
    plt.show()
    #plt.savefig('prediction_error.png', dpi=400)

def plot_prediction_idle_periods (actual_idle_periods, predicted_idle_periods):
    print (actual_idle_periods)
    print (predicted_idle_periods)

    prediction_keys = list (predicted_idle_periods.keys ())

    for prediction_key in prediction_keys:
        prediction_dist = {}
        print (prediction_key, len (predicted_idle_periods[prediction_key]))
        for prediction in predicted_idle_periods[prediction_key]:
            cpu_idle_periods = prediction[0]
            gpu_idle_periods = prediction[1]

            print (cpu_idle_periods)
            print (gpu_idle_periods)

            cpu_result = {}
            for cpu_idle_period in cpu_idle_periods:
                cpu_idle_period_dict = cpu_idle_period[2]
                for cpu_id in cpu_idle_period_dict.keys ():
                    cpu_idle_period_list = cpu_idle_period_dict[cpu_id]
                    for idle_period in cpu_idle_period_list:
                        if idle_period[2] > 0:
                            if cpu_id not in cpu_result.keys():
                                cpu_result[cpu_id] = idle_period[2]
                            else:
                                cpu_result[cpu_id] += idle_period[2]

            gpu_result = {}
            for gpu_idle_period in gpu_idle_periods:
                gpu_idle_period_dict = gpu_idle_period[2]
                for gpu_id in gpu_idle_period_dict.keys():
                    gpu_idle_period_list = gpu_idle_period_dict[gpu_id]
                    for idle_period in gpu_idle_period_list:
                        if idle_period[2] > 0:
                            if gpu_id not in gpu_result.keys():
                                gpu_result[gpu_id] = idle_period[2]
                            else:
                                gpu_result[gpu_id] += idle_period[2]

            for cpu_id in cpu_result.keys ():
                if cpu_id not in prediction_dist.keys ():
                    prediction_dist[cpu_id] = []
                prediction_dist[cpu_id].append (cpu_result[cpu_id])

            for gpu_id in gpu_result.keys():
                if gpu_id not in prediction_dist.keys():
                    prediction_dist[gpu_id] = []
                prediction_dist[gpu_id].append(gpu_result[gpu_id])

        print (prediction_dist)



def plot_prediction_sim_0 (rmanager, plot_data, prediction_times, batchsize):
    fig, axes = plt.subplots(5, 1, sharex=True)

    labels = ['stage1', 'stage2', 'stage3', 'stage4', 'stage5']

    pipelinestage_index = 0
    for pipelinestage_name in plot_data:
        phases_data = plot_data[pipelinestage_name]
        phase_index = 0
        ax = axes[pipelinestage_index ]

        #x.set_ylim (bottom=-2, top=batchsize)

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


    #plt.savefig('tmp.png', dpi=300)

    fig1, axes1 = plt.subplots(nrows=1, ncols=1)

    resourcetypeinfo = rmanager.get_resourcetype_info_all()

    print (resourcetypeinfo)

    ax = axes1

    if 'on_demand' in resourcetypeinfo.keys ():
        for resource_name in resourcetypeinfo['on_demand'].keys ():
            x_data_on_demand = [i * 3600 for i in resourcetypeinfo['on_demand'][resource_name]['count']['time']]
            y_data_on_demand = resourcetypeinfo['on_demand'][resource_name]['count']['count']

            x_data_on_demand_new = [x_data_on_demand[0]]
            y_data_on_demand_new = [y_data_on_demand[0]]

            index = 1

            while index < len (x_data_on_demand):
                x_data_on_demand_new.append (x_data_on_demand[index])
                y_data_on_demand_new.append (y_data_on_demand[index - 1])
                x_data_on_demand_new.append (x_data_on_demand[index])
                y_data_on_demand_new.append(y_data_on_demand[index])
                index += 1


            #print(resource_name, x_data, y_data)
            if resourcetypeinfo['on_demand'][resource_name]['resourcetype'] == 'CPU':
                ax.plot(x_data_on_demand_new, y_data_on_demand_new, label=resource_name, linestyle='solid')
            else:
                ax.plot (x_data_on_demand_new, y_data_on_demand_new, label=resource_name, linestyle='dashed')


    if 'spot' in resourcetypeinfo.keys ():
        for resource_name in resourcetypeinfo['spot'].keys ():
            x_data_spot = [i * 3600 for i in resourcetypeinfo['spot'][resource_name]['count']['time']]
            y_data_spot = resourcetypeinfo['spot'][resource_name]['count']['count']
            #print(resource_name, x_data, y_data)
            if resourcetypeinfo['spot'][resource_name]['resourcetype'] == 'CPU':
                ax.plot (x_data_spot, y_data_spot, label=resource_name, linestyle='dotted')
            else:
                ax.plot(x_data_spot, y_data_spot, label=resource_name, linestyle='dashdott')
    ax.legend()
    ax.set_xlabel ('Timeline (seconds)')
    ax.set_ylabel ('Count')
    fig.savefig('queue_pattern_overallocation_stable.png', dpi=300)
    fig1.savefig('resource_pattern_overallocation_stable.png', dpi=300)
    plt.show()