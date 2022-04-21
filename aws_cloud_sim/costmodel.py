import requests
from bs4 import BeautifulSoup
import pickle
import json
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class AWSCostModel:
    def __init__(self, zone, product, env):
        self.env = env
        self.zone = zone
        self.product = product
        self.spot_cpu_price_timeseries = {}
        self.spot_gpu_price_timeseries = {}
        print ('init cost model')
        self.read_on_demand_cost_model('')
        self.read_spot_cost_model('')
        print ('init cost model done')

    def get_spot_cost (self, resourcetype, computetype, bidding_price):
        if computetype == 'CPU':
            if resourcetype not in self.spot_cpu_price_timeseries:
                return None
            spot_series = self.spot_cpu_price_timeseries[resourcetype]

            current_time = int (self.env.now * 3600)

            price = spot_series.at [current_time, 'price']

            return price

        elif computetype == 'GPU':
            if resourcetype not in self.spot_gpu_price_timeseries:
                return None
            spot_series = self.spot_gpu_price_timeseries[resourcetype]

            current_time = int(self.env.now * 3600)

            price = spot_series.at[current_time, 'price']

            return price
        else:
            print (resourcetype, computetype, 'not in cost model')
            return None

    def get_on_demand_cost (self, resourcetype, computetype):
        if computetype == 'CPU':
            if resourcetype not in self.cpu_on_demand_prices:
                return None
            return float (self.cpu_on_demand_prices[resourcetype])
        elif computetype == 'GPU':
            if resourcetype not in self.gpu_on_demand_prices:
                return None
            return float (self.gpu_on_demand_prices[resourcetype])
        else:
            print (resourcetype, computetype, 'not in cost model')
            return None

    def get_on_demand_startup_time (self, resourcetype, computetype):
        return float (120/3600)

    def get_spot_startup_time (self, resourcetype, computetype):
        return float (720/3600)

    def build_ondemand_cost_model (self, prefix):
        cpu_on_demand_prices_file = open(prefix + 'aws_cloud_sim/cpu_on_demand_prices.pkl', 'rb')

        self.cpu_on_demand_prices = pickle.load(cpu_on_demand_prices_file)

        #for keys in self.cpu_on_demand_prices:
        #    print(keys, '=>', self.cpu_on_demand_prices[keys])
        cpu_on_demand_prices_file.close()

        gpu_on_demand_prices_file = open(prefix + 'aws_cloud_sim/gpu_on_demand_prices.pkl', 'rb')
        self.gpu_on_demand_prices = pickle.load(gpu_on_demand_prices_file)

        #for keys in self.gpu_on_demand_prices:
        #    print(keys, '=>', self.gpu_on_demand_prices[keys])
        gpu_on_demand_prices_file.close()

        on_demand_cost_model_pkl = open(prefix + 'aws_cloud_sim/on_demand_cost_model.pkl', 'wb')

        pickle.dump([self.cpu_on_demand_prices, self.gpu_on_demand_prices], on_demand_cost_model_pkl)

        on_demand_cost_model_pkl.close()

    def read_on_demand_cost_model (self, prefix):
        spot_cost_model_pkl = open(prefix + 'aws_cloud_sim/on_demand_cost_model.pkl', 'rb')

        [self.cpu_on_demand_prices, self.gpu_on_demand_prices] = pickle.load(spot_cost_model_pkl)

        #print (self.cpu_on_demand_prices)
        #print (self.gpu_on_demand_prices)
        #for keys in self.cpu_on_demand_prices:
        #    print(keys, '=>', self.cpu_on_demand_prices[keys])

        #for keys in self.gpu_on_demand_prices:
        #    print(keys, '=>', self.gpu_on_demand_prices[keys])

    def build_spot_cost_model (self, prefix):
        cpu_file = open(prefix + 'aws_cloud_sim/cpu_price_history')
        cpu_json = json.loads(cpu_file.read())

        gpu_file = open(prefix + 'aws_cloud_sim/gpu_price_history')
        gpu_json = json.loads(gpu_file.read())

        cpu_prices = cpu_json['SpotPriceHistory']
        gpu_prices = gpu_json['SpotPriceHistory']

        cpu_price_dict = {}
        gpu_price_dict = {}

        availability_zone = self.zone
        product_description = self.product

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
            #print(interpolated_df.head())
            #print (instance_type, 'resampling done')

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
            #print (interpolated_df.head ())
            #print(instance_type, 'resampling done')

        file_name = self.zone.replace("-", "_") + '_' + self.product.replace("/",
                                                                             "_") + '_' + 'spot_cost_model.pkl'
        spot_cost_model_pkl = open(prefix + 'aws_cloud_sim/' + file_name, 'wb')

        pickle.dump([self.spot_cpu_price_timeseries, self.spot_cpu_price_timeseries], spot_cost_model_pkl)

        spot_cost_model_pkl.close()

    def read_spot_cost_model (self, prefix):
        file_name = self.zone.replace("-", "_") + '_' + self.product.replace("/",
                                                                             "_") + '_' + 'spot_cost_model.pkl'
        spot_cost_model_pkl = open(prefix + 'aws_cloud_sim/' + file_name, 'rb')

        [self.spot_cpu_price_timeseries, self.spot_gpu_price_timeseries] = pickle.load(spot_cost_model_pkl)

        #print (self.spot_cpu_price_timeseries)
        #print (self.spot_gpu_price_timeseries)


if __name__ == "__main__":
    costmodel = AWSCostModel ('us-east-1a', 'Linux/UNIX', None)
    costmodel.build_ondemand_cost_model('../')
    costmodel.read_on_demand_cost_model('../')
    #costmodel.build_spot_cost_model('../')
    #costmodel.read_spot_cost_model('../')