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

    data_df.plot(kind='box')

    labels = ['stage 1', 'stage 2', 'stage 3', 'stage 4', 'stage 5']

    plt.xlabel ('pipelinestages')
    plt.ylabel ('Output Size (MB)')

    plt.show()

if __name__ == '__main__':
    analysis()