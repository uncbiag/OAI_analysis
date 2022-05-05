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
import statistics

matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['font.family'] = 'Times New Roman'

def exec_times ():
    exec_time_dict = {'0': [13.4, 12.74,22.43,14.63, 16.7,],
                      '1': [67.42, 86.16, 59.85],
                      '2': [180.94, 246.9,357.82, 192.02, 346.61],
                      '3': [59.96, 81.8, 44.56],
                      '4': [155.92, 199.97, 265.4, 153.72, 288.83],
                      }
    return exec_time_dict

def analysis ():
    dbfile = open('filesizes.pkl', 'rb')
    filesizes = pickle.load(dbfile)
    dbfile.close()

    for pipelinestage in filesizes:
        '''
        print (pipelinestage, len (list (filesizes[pipelinestage])))
        for filename in filesizes[pipelinestage]:
            print (filename, filesizes[pipelinestage][filename])
        '''

        temp = []
        res = dict()
        for key, val in filesizes[pipelinestage].items():
            if val not in temp:
                temp.append(val)
                res[key] = val

        filesizes[pipelinestage] = res

    final_sizes = {}

    for pipelinestage in filesizes:
        print (pipelinestage, len (list (filesizes[pipelinestage])))

        results = {}
        for filename in filesizes[pipelinestage]:
            key = filename.split('/')[6]
            if key in results:
                results[key] += float (filesizes[pipelinestage][filename]) / float (1024 * 1024)
            else:
                results[key] = float (filesizes[pipelinestage][filename]) / float (1024 * 1024)

        print (len (list (results.keys ())), results)
        final_sizes[pipelinestage] = list (results.values ())

    data_df = pd.DataFrame.from_dict(final_sizes)

    print (data_df.head())

    labels = ['stage 1', 'stage 2', 'stage 3', 'stage 4', 'stage 5']

    fig, ax = plt.subplots(1, 1)

    #ax = data_df.plot(kind='box')

    exec_times_dict = exec_times()

    #for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    #    plt.setp(bp[element], color=edge_color)

    bp1 = ax.boxplot (list (final_sizes.values()))

    for key in final_sizes:
        print (key, statistics.mean (final_sizes[key]))

    bp2 = ax.boxplot (list (exec_times_dict.values ()))

    plt.setp (bp1['medians'], color='red')
    plt.setp(bp1['boxes'], color='red')

    plt.setp(bp2['medians'], color='blue')
    plt.setp(bp2['boxes'], color='blue')

    ax.set_xticklabels (labels)

    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['output size', 'exec time'], loc='upper right')

    plt.xlabel ('pipelinestages')
    plt.ylabel ('Output Size (MB)/Exec Time (secs)')

    #ax.legend ()

    plt.savefig ('intermediate_output', dpi=400)
    plt.show ()

if __name__ == '__main__':
    analysis()