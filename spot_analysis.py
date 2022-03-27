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

def get_pricing_history ():
    url = 'https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/compute-optimized-instances.html'

    data = requests.get(url).text

    soup = BeautifulSoup(data, 'html.parser')

    table = soup.find('table', attrs={'id': 'w860aac18c13c27c27b5'})

    # print (table)

    table_head = table.find('thead')

    headers = table_head.find_all('th')

    headers = [header.text.strip() for header in headers]

    print(headers)

    table_data = []

    rows = table.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        if len(cols) <= 0:
            continue
        cols = [ele.text.strip() for ele in cols]

        table_data.append([ele for ele in cols if ele])

    print(table_data)

    cpu_df = pd.DataFrame(table_data, columns=headers)
    print(cpu_df.head())

    url = 'https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/accelerated-computing-instances.html'

    data = requests.get(url).text

    soup = BeautifulSoup(data, 'html.parser')

    table = soup.find('table', attrs={'id': 'w860aac18c13c33c19b5'})

    table_head = table.find('thead')

    headers = table_head.find_all('th')

    headers = [header.text.strip() for header in headers]

    print(headers)

    table_data = []

    rows = table.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        if len(cols) <= 0:
            continue
        cols = [ele.text.strip() for ele in cols]

        table_data.append([ele for ele in cols if ele])

    print(table_data)

    gpu_df = pd.DataFrame(table_data, columns=headers)
    print(gpu_df.head())

def analysis ():
    cpu_file = open ('cpu_price_history')
    cpu_json = json.loads(cpu_file.read())

    gpu_file = open ('gpu_price_history')
    gpu_json = json.loads(gpu_file.read())

    '''
    cpu_dict1 = {}
    cpu_dict2 = {}

    cpu_prices = cpu_json['SpotPriceHistory']

    cpu_instance_types = []

    for cpu_price in cpu_prices:
        if cpu_price['InstanceType'] not in cpu_instance_types:
            cpu_instance_types.append(cpu_price['InstanceType'])

    cpu_group1 = cpu_instance_types[0:int(len(cpu_instance_types) / 2)]
    cpu_group2 = cpu_instance_types[int(len(cpu_instance_types) / 2):]

    print (len (cpu_group1), cpu_group1)
    print (len (cpu_group2), cpu_group2)

    cpu_instances1 = []
    cpu_zones1 = []
    cpu_os1 = []
    cpu_spot_prices1 = []

    cpu_instances2 = []
    cpu_zones2 = []
    cpu_os2 = []
    cpu_spot_prices2 = []


    for cpu_price in cpu_prices:
        if cpu_price['InstanceType'] in cpu_group1:
            cpu_instances1.append(cpu_price['InstanceType'])
            cpu_zones1.append(cpu_price['AvailabilityZone'])
            cpu_os1.append (cpu_price['ProductDescription'])
            cpu_spot_prices1.append(float(cpu_price['SpotPrice']))
        else:
            cpu_instances2.append(cpu_price['InstanceType'])
            cpu_zones2.append(cpu_price['AvailabilityZone'])
            cpu_os2.append(cpu_price['ProductDescription'])
            cpu_spot_prices2.append(float(cpu_price['SpotPrice']))

    cpu_dict1['instance'] = cpu_instances1
    cpu_dict1['zone'] = cpu_zones1
    cpu_dict1['os'] = cpu_os1
    cpu_dict1['price'] = cpu_spot_prices1

    cpu_dict2['instance'] = cpu_instances2
    cpu_dict2['zone'] = cpu_zones2
    cpu_dict2['os'] = cpu_os2
    cpu_dict2['price'] = cpu_spot_prices2

    cpu_df1 = pd.DataFrame.from_dict(cpu_dict1)
    cpu_df2 = pd.DataFrame.from_dict(cpu_dict2)

    print (cpu_df1.head())
    print (cpu_df2.head())

    f, (ax1, ax2) = plt.subplots(2, 1)
    f.suptitle('CPU Instance Price Distribution over 2 months', fontsize=14)

    sns.boxplot(x="instance", y="price", hue="os",
                data=cpu_df1, ax=ax1)
    ax1.set_xlabel("CPU Instances", size=12, alpha=0.8)
    ax1.set_ylabel("Price/hr($)", size=12, alpha=0.8)
    ax1.set_xticklabels (cpu_group1, rotation = 45)

    sns.boxplot(x="instance", y="price", hue="os",
                data=cpu_df2, ax=ax2)
    ax2.set_xlabel("CPU Instances", size=12, alpha=0.8)
    ax2.set_ylabel("Price/hr($)", size=12, alpha=0.8)
    ax2.set_xticklabels(cpu_group2, rotation=45)


    l = plt.legend(loc='best', title='CPU Instance Groups')
    '''

    #g = sns.FacetGrid(cpu_df, col='instance', hue='os',)
    #g.map_dataframe(sns.scatterplot, x="zone", y="price", alpha=0.5,)

    gpu_dict1 = {}
    gpu_dict2 = {}

    gpu_prices = gpu_json['SpotPriceHistory']

    gpu_instance_types = []

    for gpu_price in gpu_prices:
        if gpu_price['InstanceType'] not in gpu_instance_types:
            gpu_instance_types.append(gpu_price['InstanceType'])

    gpu_group1 = gpu_instance_types[0:int(len(gpu_instance_types) / 2)]
    gpu_group2 = gpu_instance_types[int(len(gpu_instance_types) / 2):]

    print (len (gpu_group1), gpu_group1)
    print (len (gpu_group2), gpu_group2)

    gpu_instances1 = []
    gpu_zones1 = []
    gpu_os1 = []
    gpu_spot_prices1 = []

    gpu_instances2 = []
    gpu_zones2 = []
    gpu_os2 = []
    gpu_spot_prices2 = []


    for gpu_price in gpu_prices:
        if gpu_price['InstanceType'] in gpu_group1:
            gpu_instances1.append(gpu_price['InstanceType'])
            gpu_zones1.append(gpu_price['AvailabilityZone'])
            gpu_os1.append (gpu_price['ProductDescription'])
            gpu_spot_prices1.append(float(gpu_price['SpotPrice']))
        else:
            gpu_instances2.append(gpu_price['InstanceType'])
            gpu_zones2.append(gpu_price['AvailabilityZone'])
            gpu_os2.append(gpu_price['ProductDescription'])
            gpu_spot_prices2.append(float(gpu_price['SpotPrice']))

    gpu_dict1['instance'] = gpu_instances1
    gpu_dict1['zone'] = gpu_zones1
    gpu_dict1['os'] = gpu_os1
    gpu_dict1['price'] = gpu_spot_prices1

    gpu_dict2['instance'] = gpu_instances2
    gpu_dict2['zone'] = gpu_zones2
    gpu_dict2['os'] = gpu_os2
    gpu_dict2['price'] = gpu_spot_prices2

    gpu_df1 = pd.DataFrame.from_dict(gpu_dict1)
    gpu_df2 = pd.DataFrame.from_dict(gpu_dict2)

    print (gpu_df1.head())
    print (gpu_df2.head())

    f, (ax1, ax2) = plt.subplots(2, 1)
    f.suptitle('GPU Instance Price Distribution over 2 months', fontsize=14)

    sns.boxplot(x="instance", y="price", hue="os",
                data=gpu_df1, ax=ax1)
    ax1.set_xlabel("GPU Instances", size=12, alpha=0.8)
    ax1.set_ylabel("Price/hr($)", size=12, alpha=0.8)
    ax1.set_xticklabels (gpu_group1, rotation = 45)

    sns.boxplot(x="instance", y="price", hue="os",
                data=gpu_df2, ax=ax2)
    ax2.set_xlabel("GPU Instances", size=12, alpha=0.8)
    ax2.set_ylabel("Price/hr($)", size=12, alpha=0.8)
    ax2.set_xticklabels(gpu_group2, rotation=45)


    l = plt.legend(loc='best', title='GPU Instance Groups')

    plt.show ()
    '''

    for cpu_price in cpu_prices:
        print (cpu_price)
        if cpu_price['InstanceType'] not in cpu_dict:
            cpu_dict[cpu_price['InstanceType']] = {}
            cpu_dict[cpu_price['InstanceType']][cpu_price['ProductDescription']] = []
            cpu_dict[cpu_price['InstanceType']][cpu_price['ProductDescription']].append (cpu_price['SpotPrice'])
        elif cpu_price['ProductDescription'] not in cpu_dict[cpu_price['InstanceType']]:
            cpu_dict[cpu_price['InstanceType']][cpu_price['ProductDescription']] = []
            cpu_dict[cpu_price['InstanceType']][cpu_price['ProductDescription']].append(cpu_price['SpotPrice'])
        else:
            cpu_dict[cpu_price['InstanceType']][cpu_price['ProductDescription']].append(cpu_price['SpotPrice'])

    sns.boxplot(x="InstanceType", y="alcohol", hue="wine_type",
                data=cpu_dict, palette={"red": "#FF9999", "white": "white"}, ax=ax1)

    '''
    '''
    for gpu_price in gpu_prices:
        print(gpu_price)
        if gpu_price['InstanceType'] not in gpu_dict:
            gpu_dict[gpu_price['InstanceType']] = {}
            gpu_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']] = {}
            gpu_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']][gpu_price['ProductDescription']] = {}
            gpu_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']][gpu_price['ProductDescription']][gpu_price['Timestamp']] = gpu_price['SpotPrice']
        elif gpu_price['AvailabilityZone'] not in gpu_price['InstanceType']:
            gpu_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']] = {}
            gpu_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']][gpu_price['ProductDescription']] = {}
            gpu_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']][gpu_price['ProductDescription']][gpu_price['Timestamp']] = gpu_price['SpotPrice']
        elif gpu_price['ProductDescription'] not in gpu_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']]:
            gpu_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']][gpu_price['ProductDescription']] = {}
            gpu_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']][gpu_price['ProductDescription']][gpu_price['Timestamp']] = gpu_price['SpotPrice']
        else:
            gpu_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']][gpu_price['ProductDescription']][gpu_price['Timestamp']] = gpu_price['SpotPrice']
    '''

if __name__ == "__main__":
    analysis()