import requests
from bs4 import BeautifulSoup
import pickle
import json
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class AWSCostModel:
    def __init__(self, region, product):
        self.spot_cpu_price_timeseries = {}
        self.spot_gpu_price_timeseries = {}
        self.build_ondemand_cost_model ()
        self.build_spot_cost_model (region, product)

    def build_ondemand_cost_model (self):
        cpu_on_demand_prices_file = open('cpu_on_demand_prices.pkl', 'rb')
        self.cpu_on_demand_prices = pickle.load(cpu_on_demand_prices_file)

        for keys in self.cpu_on_demand_prices:
            print(keys, '=>', self.cpu_on_demand_prices[keys])
        cpu_on_demand_prices_file.close()

        gpu_on_demand_prices_file = open('gpu_on_demand_prices.pkl', 'rb')
        self.gpu_on_demand_prices = pickle.load(gpu_on_demand_prices_file)

        for keys in self.gpu_on_demand_prices:
            print(keys, '=>', self.gpu_on_demand_prices[keys])
        gpu_on_demand_prices_file.close()

    def build_spot_cost_model (self, availability_zone, product_description):
        cpu_file = open('cpu_price_history')
        cpu_json = json.loads(cpu_file.read())

        gpu_file = open('gpu_price_history')
        gpu_json = json.loads(gpu_file.read())

        cpu_prices = cpu_json['SpotPriceHistory']
        gpu_prices = gpu_json['SpotPriceHistory']

        cpu_price_dict = {}
        gpu_price_dict = {}

        for cpu_price in cpu_prices:
            if cpu_price['ProductDescription'] != product_description or cpu_price['AvailabilityZone'] != availability_zone:
                continue
            if cpu_price['InstanceType'] not in cpu_price_dict:
                cpu_price_dict[cpu_price['InstanceType']] = {}
                if cpu_price['AvailabilityZone'] not in cpu_price_dict[cpu_price['InstanceType']]:
                    cpu_price_dict[cpu_price['InstanceType']][cpu_price['AvailabilityZone']] = {}
                    if cpu_price['ProductDescription'] not in cpu_price_dict[cpu_price['InstanceType']][
                        cpu_price['AvailabilityZone']]:
                        cpu_price_dict[cpu_price['InstanceType']][cpu_price['AvailabilityZone']][
                            cpu_price['ProductDescription']] = {}
                        cpu_price_dict[cpu_price['InstanceType']][cpu_price['AvailabilityZone']][
                            cpu_price['ProductDescription']][cpu_price['Timestamp']] = cpu_price['SpotPrice']
                    else:
                        cpu_price_dict[cpu_price['InstanceType']][cpu_price['AvailabilityZone']][
                            cpu_price['ProductDescription']][cpu_price['Timestamp']] = cpu_price['SpotPrice']
                else:
                    if cpu_price['ProductDescription'] not in cpu_price_dict[cpu_price['InstanceType']][
                        cpu_price['AvailabilityZone']]:
                        cpu_price_dict[cpu_price['InstanceType']][cpu_price['AvailabilityZone']][
                            cpu_price['ProductDescription']] = {}
                        cpu_price_dict[cpu_price['InstanceType']][cpu_price['AvailabilityZone']][
                            cpu_price['ProductDescription']][cpu_price['Timestamp']] = cpu_price['SpotPrice']
                    else:
                        cpu_price_dict[cpu_price['InstanceType']][cpu_price['AvailabilityZone']][
                            cpu_price['ProductDescription']][cpu_price['Timestamp']] = cpu_price['SpotPrice']
            else:
                if cpu_price['AvailabilityZone'] not in cpu_price_dict[cpu_price['InstanceType']]:
                    cpu_price_dict[cpu_price['InstanceType']][cpu_price['AvailabilityZone']] = {}
                    if cpu_price['ProductDescription'] not in cpu_price_dict[cpu_price['InstanceType']][
                        cpu_price['AvailabilityZone']]:
                        cpu_price_dict[cpu_price['InstanceType']][cpu_price['AvailabilityZone']][
                            cpu_price['ProductDescription']] = {}
                        cpu_price_dict[cpu_price['InstanceType']][cpu_price['AvailabilityZone']][
                            cpu_price['ProductDescription']][cpu_price['Timestamp']] = cpu_price['SpotPrice']
                    else:
                        cpu_price_dict[cpu_price['InstanceType']][cpu_price['AvailabilityZone']][
                            cpu_price['ProductDescription']][cpu_price['Timestamp']] = cpu_price['SpotPrice']
                else:
                    if cpu_price['ProductDescription'] not in cpu_price_dict[cpu_price['InstanceType']][
                        cpu_price['AvailabilityZone']]:
                        cpu_price_dict[cpu_price['InstanceType']][cpu_price['AvailabilityZone']][
                            cpu_price['ProductDescription']] = {}
                        cpu_price_dict[cpu_price['InstanceType']][cpu_price['AvailabilityZone']][
                            cpu_price['ProductDescription']][cpu_price['Timestamp']] = cpu_price['SpotPrice']
                    else:
                        cpu_price_dict[cpu_price['InstanceType']][cpu_price['AvailabilityZone']][
                            cpu_price['ProductDescription']][cpu_price['Timestamp']] = cpu_price['SpotPrice']

        for gpu_price in gpu_prices:
            if gpu_price['ProductDescription'] != product_description or gpu_price['AvailabilityZone'] != availability_zone:
                continue
            if gpu_price['InstanceType'] not in gpu_price_dict:
                gpu_price_dict[gpu_price['InstanceType']] = {}
                if gpu_price['AvailabilityZone'] not in gpu_price_dict[gpu_price['InstanceType']]:
                    gpu_price_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']] = {}
                    if gpu_price['ProductDescription'] not in gpu_price_dict[gpu_price['InstanceType']][
                        gpu_price['AvailabilityZone']]:
                        gpu_price_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']][
                            gpu_price['ProductDescription']] = {}
                        gpu_price_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']][
                            gpu_price['ProductDescription']][gpu_price['Timestamp']] = gpu_price['SpotPrice']
                    else:
                        gpu_price_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']][
                            gpu_price['ProductDescription']][gpu_price['Timestamp']] = gpu_price['SpotPrice']
                else:
                    if gpu_price['ProductDescription'] not in gpu_price_dict[gpu_price['InstanceType']][
                        gpu_price['AvailabilityZone']]:
                        gpu_price_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']][
                            gpu_price['ProductDescription']] = {}
                        gpu_price_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']][
                            gpu_price['ProductDescription']][gpu_price['Timestamp']] = gpu_price['SpotPrice']
                    else:
                        gpu_price_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']][
                            gpu_price['ProductDescription']][gpu_price['Timestamp']] = gpu_price['SpotPrice']
            else:
                if gpu_price['AvailabilityZone'] not in gpu_price_dict[gpu_price['InstanceType']]:
                    gpu_price_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']] = {}
                    if gpu_price['ProductDescription'] not in gpu_price_dict[gpu_price['InstanceType']][
                        gpu_price['AvailabilityZone']]:
                        gpu_price_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']][
                            gpu_price['ProductDescription']] = {}
                        gpu_price_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']][
                            gpu_price['ProductDescription']][gpu_price['Timestamp']] = gpu_price['SpotPrice']
                    else:
                        gpu_price_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']][
                            gpu_price['ProductDescription']][gpu_price['Timestamp']] = gpu_price['SpotPrice']
                else:
                    if gpu_price['ProductDescription'] not in gpu_price_dict[gpu_price['InstanceType']][
                        gpu_price['AvailabilityZone']]:
                        gpu_price_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']][
                            gpu_price['ProductDescription']] = {}
                        gpu_price_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']][
                            gpu_price['ProductDescription']][gpu_price['Timestamp']] = gpu_price['SpotPrice']
                    else:
                        gpu_price_dict[gpu_price['InstanceType']][gpu_price['AvailabilityZone']][
                            gpu_price['ProductDescription']][gpu_price['Timestamp']] = gpu_price['SpotPrice']

        gpu_instance_types_to_be_dropped = []

        for instance_type in gpu_price_dict.keys():
            if instance_type in gpu_instance_types_to_be_dropped:
                self.gpu_on_demand_prices.pop (instance_type)
                continue
            zone_product_data = dict(
                sorted(gpu_price_dict[instance_type][availability_zone][product_description].items()))
            timestampstr_list = list(zone_product_data.keys())
            pricestr_list = list(zone_product_data.values())
            timestamp_list = []
            price_list = []
            for timestamp_str in timestampstr_list:
                timestamp_datetime = datetime.fromisoformat(timestamp_str)
                timestamp_list.append(timestamp_datetime)
            for pricestr in pricestr_list:
                price_list.append(float(pricestr))

            data_dict = {'timestamp': timestamp_list, 'price': price_list}
            data_df = pd.DataFrame(data_dict)
            data_df.set_index('timestamp', inplace=True)
            resampled_data = data_df.resample('S')
            interpolated_df = resampled_data.interpolate(method='linear')

            self.spot_gpu_price_timeseries[instance_type] = interpolated_df
            print (instance_type, 'resampling done')

        cpu_instance_types_to_be_dropped = ['c6a.2xlarge', 'c6a.12xlarge', 'c6a.32xlarge', 'c6a.large', 'c6a.48xlarge', 'c6a.4xlarge', 'c6a.8xlarge', 'c6a.xlarge', 'c6a.24xlarge', 'c6a.16xlarge']

        for instance_type in cpu_price_dict.keys():
            if instance_type in cpu_instance_types_to_be_dropped:
                self.cpu_on_demand_prices.pop (instance_type)
                continue
            zone_product_data = dict (sorted (cpu_price_dict[instance_type][availability_zone][product_description].items ()))
            timestampstr_list = list (zone_product_data.keys ())
            pricestr_list = list (zone_product_data.values ())
            timestamp_list = []
            price_list = []
            for timestamp_str in timestampstr_list:
                timestamp_datetime = datetime.fromisoformat(timestamp_str)
                timestamp_list.append(timestamp_datetime)
            for pricestr in pricestr_list:
                price_list.append (float (pricestr))

            data_dict = {'timestamp':timestamp_list, 'price':price_list}
            data_df = pd.DataFrame (data_dict)
            data_df.set_index('timestamp', inplace=True)
            resampled_data = data_df.resample('S')
            interpolated_df = resampled_data.interpolate(method='linear')

            self.spot_cpu_price_timeseries[instance_type] = interpolated_df
            print(instance_type, 'resampling done')


if __name__ == "__main__":
    costmodel = AWSCostModel ('us-east-1a', 'Linux/UNIX')