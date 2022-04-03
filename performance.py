import requests
from bs4 import BeautifulSoup
from pprint import pprint
import pandas as pd
import json
import matplotlib.pyplot as plt
from pandas.plotting import table
import matplotlib as mpl
import numpy as np
import seaborn as sns
import matplotlib
import pickle

matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['font.family'] = 'Times New Roman'

base_cpu_cost = {}
base_gpu_cost = {}
base_execution_time = {}

reconfiguration_down_cpu_cost = {}
reconfiguration_down_gpu_cost = {}
reconfiguration_down_execution_time = {}

reconfiguration_up_down_overallocation_cpu_cost = {}
reconfiguration_up_down_overallocation_gpu_cost = {}
reconfiguration_up_down_overallocation_execution_time = {}

reconfiguration_up_down_underallocation_cpu_cost = {}
reconfiguration_up_down_underallocation_gpu_cost = {}
reconfiguration_up_down_underallocation_execution_time = {}

reconfiguration_time_delta = None

def add_performance_data (algo, cpu_cost, gpu_cost, execution_time, reconfiguration_time):
    global  reconfiguration_time_delta
    print (algo, cpu_cost, gpu_cost, execution_time, reconfiguration_time)
    if algo == 'base':
        key = len (list(base_cpu_cost.keys()))
        base_cpu_cost[str(key)] = cpu_cost
        base_gpu_cost[str(key)] = gpu_cost
        base_execution_time[str(key)] = execution_time
    elif algo == 'down':
        key = len(list(reconfiguration_down_cpu_cost.keys()))
        reconfiguration_down_cpu_cost[str(key)] = cpu_cost
        reconfiguration_down_gpu_cost[str(key)] = gpu_cost
        reconfiguration_down_execution_time[str(key)] = execution_time
    elif algo == 'overallocation':
        key = len (list(reconfiguration_up_down_overallocation_cpu_cost))
        reconfiguration_up_down_overallocation_cpu_cost[str(key)] = cpu_cost
        reconfiguration_up_down_overallocation_gpu_cost[str(key)] = gpu_cost
        reconfiguration_up_down_overallocation_execution_time[str(key)] = execution_time
    elif algo == 'underallocation':
        key = len(list(reconfiguration_up_down_underallocation_cpu_cost))
        reconfiguration_up_down_underallocation_cpu_cost[str(key)] = cpu_cost
        reconfiguration_up_down_underallocation_gpu_cost[str(key)] = gpu_cost
        reconfiguration_up_down_underallocation_execution_time[str(key)] = execution_time

    reconfiguration_time_delta = reconfiguration_time

def store_performance_data (algo):
    global reconfiguration_time_delta
    print (reconfiguration_time_delta)
    data = []
    if algo == 'base':
        data.append(base_cpu_cost)
        data.append(base_gpu_cost)
        data.append(base_execution_time)
        dbfile = open('performance_database_base_'+str(reconfiguration_time_delta), 'wb')
    elif algo == 'down':
        data.append(reconfiguration_down_cpu_cost)
        data.append(reconfiguration_down_gpu_cost)
        data.append(reconfiguration_down_execution_time)
        dbfile = open('performance_database_down_'+str(reconfiguration_time_delta), 'wb')
    elif algo == 'overallocation':
        data.append(reconfiguration_up_down_overallocation_cpu_cost)
        data.append(reconfiguration_up_down_overallocation_gpu_cost)
        data.append(reconfiguration_up_down_overallocation_execution_time)
        dbfile = open('performance_database_overallocation_'+str(reconfiguration_time_delta), 'wb')
    elif algo == 'underallocation':
        data.append(reconfiguration_up_down_underallocation_cpu_cost)
        data.append(reconfiguration_up_down_underallocation_gpu_cost)
        data.append(reconfiguration_up_down_underallocation_execution_time)
        dbfile = open('performance_database_underallocation_'+str(reconfiguration_time_delta), 'wb')

    data.append(reconfiguration_time_delta)

    pickle.dump(data, dbfile)
    dbfile.close()

def analysis3 ():
    reconfiguration_time_delta_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    cpu_cost = {}
    gpu_cost = {}
    execution_time = {}

    cpu_cost_data = []
    gpu_cost_data = []
    execution_time_data = []

    for reconfiguration_time in reconfiguration_time_delta_list:
        dbfile = open('performance_database_overallocation_'+str(reconfiguration_time), 'rb')
        data = pickle.load(dbfile)
        cpu_cost[str(reconfiguration_time)] = data[0]
        gpu_cost[str(reconfiguration_time)] = data[1]
        execution_time[str(reconfiguration_time)] = data[2]
        cpu_cost_data.append(list (data[0].values ()))
        gpu_cost_data.append(list (data[1].values ()))
        execution_time_data.append(list (data[2].values ()))

    fig1, axes1 = plt.subplots(nrows=1, ncols=1)

    bp = axes1.boxplot(cpu_cost_data)

    fig2, axes2 = plt.subplots(nrows=1, ncols=1)

    bp = axes2.boxplot(gpu_cost_data)

    fig3, axes3 = plt.subplots(nrows=1, ncols=1)

    bp = axes3.boxplot(execution_time_data)

    plt.show()

def analysis2 ():
    group_data = []
    base_data = []
    down_data = []
    overallocation_data = []
    underallocation_data = []

    dbfile = open('performance_database_base', 'rb')
    data = pickle.load(dbfile)
    base_cpu_cost = data[0]
    base_gpu_cost = data[1]
    base_execution_time = data[2]

    dbfile = open('performance_database_down', 'rb')
    data = pickle.load(dbfile)
    reconfiguration_down_cpu_cost = data[0]
    reconfiguration_down_gpu_cost = data[1]
    reconfiguration_down_execution_time = data[2]

    dbfile = open('performance_database_overallocation', 'rb')
    data = pickle.load(dbfile)
    reconfiguration_up_down_overallocation_cpu_cost = data[0]
    reconfiguration_up_down_overallocation_gpu_cost = data[1]
    reconfiguration_up_down_overallocation_execution_time = data[2]

    dbfile = open('performance_database_underallocation', 'rb')
    data = pickle.load(dbfile)
    reconfiguration_up_down_underallocation_cpu_cost = data[0]
    reconfiguration_up_down_underallocation_gpu_cost = data[1]
    reconfiguration_up_down_underallocation_execution_time = data[2]


    for key in base_cpu_cost.keys ():
        group_data.append ('CPU Cost')
        base_data.append (base_cpu_cost[key])
        down_data.append (reconfiguration_down_cpu_cost[key])
        overallocation_data.append (reconfiguration_up_down_overallocation_cpu_cost[key])
        underallocation_data.append (reconfiguration_up_down_underallocation_cpu_cost[key])

    for key in base_gpu_cost.keys ():
        group_data.append ('GPU Cost')
        base_data.append (base_gpu_cost[key])
        down_data.append (reconfiguration_down_gpu_cost[key])
        overallocation_data.append (reconfiguration_up_down_overallocation_gpu_cost[key])
        underallocation_data.append (reconfiguration_up_down_underallocation_gpu_cost[key])

    for key in base_cpu_cost.keys ():
        group_data.append ('Execution Time')
        base_data.append (base_execution_time[key])
        down_data.append (reconfiguration_down_execution_time[key])
        overallocation_data.append (reconfiguration_up_down_overallocation_execution_time[key])
        underallocation_data.append (reconfiguration_up_down_underallocation_execution_time[key])

    df = pd.DataFrame({'Group':group_data,\
                  'Base':base_data,'Down':down_data, 'Overallocation':overallocation_data, 'Underallocation':underallocation_data})
    df = df[['Group','Base','Down', 'Overallocation', 'Underallocation']]
    print (df)

    dd = pd.melt(df, id_vars=['Group'], value_vars=['Base', 'Down', 'Overallocation', 'Underallocation'], var_name='Algorithms')
    sns.boxplot(x='Group', y='value', data=dd, hue='Algorithms')

    plt.show()

def analysis1 ():
    dbfile = open('performance_database', 'rb')
    data = pickle.load(dbfile)

    group_data = []
    base_data = []
    down_data = []
    overallocation_data = []
    underallocation_data = []

    base_cpu_cost = data[0]
    base_gpu_cost = data[1]
    base_execution_time = data[2]
    reconfiguration_down_cpu_cost = data[3]
    reconfiguration_down_gpu_cost = data[4]
    reconfiguration_down_execution_time = data[5]
    reconfiguration_up_down_overallocation_cpu_cost = data[6]
    reconfiguration_up_down_overallocation_gpu_cost = data[7]
    reconfiguration_up_down_overallocation_execution_time = data[8]
    reconfiguration_up_down_underallocation_cpu_cost = data[9]
    reconfiguration_up_down_underallocation_gpu_cost = data[10]
    reconfiguration_up_down_underallocation_execution_time = data[11]


    for key in base_cpu_cost.keys ():
        group_data.append ('CPU Cost')
        base_data.append (base_cpu_cost[key])
        down_data.append (reconfiguration_down_cpu_cost[key])
        overallocation_data.append (reconfiguration_up_down_overallocation_cpu_cost[key])
        underallocation_data.append (reconfiguration_up_down_underallocation_cpu_cost[key])

    for key in base_gpu_cost.keys ():
        group_data.append ('GPU Cost')
        base_data.append (base_gpu_cost[key])
        down_data.append (reconfiguration_down_gpu_cost[key])
        overallocation_data.append (reconfiguration_up_down_overallocation_gpu_cost[key])
        underallocation_data.append (reconfiguration_up_down_underallocation_gpu_cost[key])

    for key in base_cpu_cost.keys ():
        group_data.append ('Execution Time')
        base_data.append (base_execution_time[key])
        down_data.append (reconfiguration_down_execution_time[key])
        overallocation_data.append (reconfiguration_up_down_overallocation_execution_time[key])
        underallocation_data.append (reconfiguration_up_down_underallocation_execution_time[key])

    df = pd.DataFrame({'Group':group_data,\
                  'Base':base_data,'Down':down_data, 'Overallocation':overallocation_data, 'Underallocation':underallocation_data})
    df = df[['Group','Base','Down', 'Overallocation', 'Underallocation']]
    print (df)

    dd = pd.melt(df, id_vars=['Group'], value_vars=['Base', 'Down', 'Overallocation', 'Underallocation'], var_name='Algorithms')
    sns.boxplot(x='Group', y='value', data=dd, hue='Algorithms')

    plt.show()

def plot_prediction_performance ():
    print ('plot prediction_performance ()')
    print (base_execution_time, base_cpu_cost, base_gpu_cost)
    print (reconfiguration_down_execution_time, reconfiguration_down_cpu_cost, reconfiguration_down_gpu_cost)
    print (reconfiguration_up_down_overallocation_execution_time, reconfiguration_up_down_overallocation_cpu_cost, reconfiguration_up_down_overallocation_gpu_cost)
    print (reconfiguration_up_down_underallocation_execution_time, reconfiguration_up_down_underallocation_cpu_cost, reconfiguration_up_down_underallocation_gpu_cost)

def analysis ():
    base_config_data = [
        [18.840708333335115, 36.331944444447885, 5.190277777778269],
        [18.99195833333514, 36.62361111111459, 5.2319444444449426],
        [19.259166666668516, 37.13888888889245, 5.305555555556065],
        [19.314625000001858, 37.24583333333691, 5.325083333333384],
        [19.32975000000186, 37.27500000000359, 5.325000000000513],
        [19.03229166666848, 36.70138888889238, 5.243055555556055],
        [19.470916666668547, 37.54722222222585, 5.363888888889408],
        [19.112958333335158, 36.85694444444796, 5.265277777778281],
        [18.89616666666846, 36.43888888889234, 5.2055555555560495],
        [18.825583333335114, 36.30277777778121, 5.186111111111602]
    ]

    reconfig_data = [
        [18.426291666668387, 11.850138888889614, 5.7555555555561355],
        [18.362569444446155, 11.78513888888964, 5.865277777778375],
        [18.650000000001754, 13.307361111112066, 5.85694444444504],
        [18.50888888889062, 12.318888888889607, 5.688888888889458],
        [18.64777777777953, 13.119444444445305, 5.609722222222779],
        [18.18325000000168, 11.752916666667414, 5.844444444445038],
        [18.592361111112858, 13.780000000000992, 5.219444444444941],
        [18.13062500000167, 12.33125000000078, 5.513888888889431],
        [18.5830555555573, 12.314722222222944, 5.752777777778357],
        [18.348166666668398, 11.36222222222298, 5.956944444445056]
    ]

    base_cpu_cost_data = []
    base_gpu_cost_data = []
    base_execution_time_data = []

    for sample in base_config_data:
        base_cpu_cost_data.append(sample[0])
        base_gpu_cost_data.append(sample[1])
        base_execution_time_data.append(sample[2])


    reconfig_cpu_cost_data = []
    reconfig_gpu_cost_data = []
    reconfig_execution_time_data = []

    for sample in reconfig_data:
        reconfig_cpu_cost_data.append(sample[0])
        reconfig_gpu_cost_data.append(sample[1])
        reconfig_execution_time_data.append(sample[2])

    base_data = {}

    base_data['Base CPU Cost'] = base_cpu_cost_data
    base_data['Reconfig CPU Cost'] = reconfig_cpu_cost_data
    base_data['Base GPU Cost'] = base_gpu_cost_data
    base_data['Reconfig GPU Cost'] = reconfig_gpu_cost_data
    base_data['Base Exec Time'] = base_execution_time_data
    base_data['Reconfig Exec Time'] = reconfig_execution_time_data


    base_data_df = pd.DataFrame.from_dict(base_data)

    print (len (base_data_df.index))

    sns.boxplot(data=base_data_df)

    plt.ylabel ('Exec Time (Hours) Or Cost($)')

    plt.show()


if __name__ == "__main__":
    analysis3()