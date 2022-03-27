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

matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['font.family'] = 'Times New Roman'

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
    analysis()