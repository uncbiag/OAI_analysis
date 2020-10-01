#!/usr/bin/env python3.6
"""
This script test the results of thickness computation by comparing them with manual measurements

Created by zhenlinx on 9/17/19
"""
import math
import pandas as pd
import numpy as np
from data.OAI_data import OAIData, OAIImage, OAIPatients
from misc.str_ops import read_pid_from_file
from tqdm import tqdm
import SimpleITK as sitk
import os

def get_seg_volume(seg_file:str)->float:
    """
    Get the segmentation volume in mm^3 (could be a probability map).
    If the given file does not exists, return -1.
    :param seg_file: file path of segmentation
    :return:
    """


    if os.path.isfile(seg_file):
        seg = sitk.ReadImage(seg_file)
        seg_np = sitk.GetArrayViewFromImage(seg)
        spacing = seg.GetSpacing()
        volume = np.sum(seg_np) * spacing[0] * spacing[1] * spacing[2]
    else:
        volume = -1
    return volume

def _match_image_date_to_chart_date(chart_date:str, image_date:int):
    """
    Match two date, if their gap is less than 60 days return True
    :param chart_date: date in format "06/20/2005"
    :param image_date: date in format "20050620"
    :return:
    """
    from datetime import datetime
    image_date = str(image_date)
    image_date = datetime.strptime(image_date, '%Y%m%d')
    chart_date = datetime.strptime(chart_date, '%m/%d/%Y')
    return abs(image_date - chart_date).days <= 60

def match_image_date_to_chart_dates(chart_dates:pd.Series, image_date:int):
    res = pd.Series(index=chart_dates.index)
    for ind, date in chart_dates.items():
        if (type(date) is float and math.isnan(date)) or (type(date) is str and len(date) > 10):
            res[ind] = False
        else:
            res[ind] = _match_image_date_to_chart_date(date, image_date)
    return res

def process_score_sets(nums):
    """get the median of non-NaN values"""
    values = []
    for n in nums:
        if not math.isnan(n):
            values.append(n)
    if values == []:
        return -1
    else:
        return np.median(values)

# set-up OAI database
use_nifti=False
OAI_data_sheet = "data/SEG_3D_DESS_6visits.csv"
OAI_data = OAIData(OAI_data_sheet)
# we do not create the directories here, as we want to do this on the fly
task_name = None if use_nifti else 'avsm'
OAI_data.set_processed_data_paths_without_creating_image_directories("/data/raid1/oai_analysis_results", task_name=task_name)

# set-up OAI patients data
patients_ASCII_file_path = "data/Enrollees.txt"
oai_patients = OAIPatients(patients_ASCII_file_path)
pids = read_pid_from_file("/data/raid1/zhenlinx/pid_600.txt")
analysis_patient = list(OAI_data.patient_set & set(pids))

# filter images
analysis_images = OAI_data.get_images(patient_id=analysis_patient, part="LEFT_KNEE")

# d = pandas.read_pickle('thickness_results.pkl')
# fc = np.load('thickness_results_femoral_cartilage.npz', allow_pickle=True)['data']
# tc = np.load('thickness_results_tibial_cartilage.npz', allow_pickle=True)['data']

# get manual evaluations
file = '/playpen-raid/zhenlinx/Data/OAI_raw/OAINonImaging/kmriqcart01.txt'
chart = pd.read_csv(file, sep='\t')

# FC_volumes_manual = {}
# TC_volumes_manual = {}
#
# FC_volumes_model = {}
# TC_volumes_model = {}


volume_file = 'volumes.csv'
volume_df = pd.read_csv(volume_file).iloc[1:]


FC_volumes_manual = []
FC_volumes_model = []
TC_volumes_manual = []
TC_volumes_model = []

for i, row in volume_df.iterrows():
    if row['TC_volumes_manual'] >= 0:
        TC_volumes_manual.append(row['TC_volumes_manual'])
        TC_volumes_model.append(row['TC_volumes_model'])
    if row['FC_volumes_manual'] >= 0:
        FC_volumes_manual.append(row['FC_volumes_manual'])
        FC_volumes_model.append(row['FC_volumes_model'])

import matplotlib.pyplot as plt
f, axes = plt.subplots(1,2, figsize=(10, 5))
axes[0].plot(TC_volumes_manual, TC_volumes_model, 'b.')
axes[0].set_xlim(2000, 9000)
axes[0].set_ylim(2000, 9000)
axes[0].set_aspect('equal')
axes[0].set_xlabel('TC manual')
axes[0].set_ylabel('TC model')
axes[1].plot(FC_volumes_manual, FC_volumes_model, 'b.')
axes[1].set_xlim(7500, 20000)
axes[1].set_ylim(7500, 20000)
axes[1].set_aspect('equal')
axes[1].set_xlabel('FC manual')
axes[1].set_ylabel('FC model')

plt.show()

# if os.path.isfile(volume_file):
#     volume_df = pd.read_csv(volume_file)
#     t = tqdm(analysis_images)
#     for image in t:
#         FC_volume_model = volume_df.loc[(volume_df['bar_code'] == image.bar_code)]['FC_volumes_model'].iloc[0]
#         TC_volume_model = volume_df.loc[(volume_df['bar_code'] == image.bar_code)]['TC_volumes_model'].iloc[0]
#
#         image_records = chart.loc[(chart['src_subject_id'] == image.patient_id) &
#                                   (match_image_date_to_chart_dates(chart['interview_date'], image.study_date))]
#
#         FC_volume_manual = process_score_sets(image_records['wfwvol'].values)
#         TC_volume_manual = process_score_sets(image_records['wmtvcl'].values) + \
#                            process_score_sets(image_records['wltvcl'].values)
#
#         if image.bar_code in volume_df['bar_code'].values:
#             volume_df.loc[(volume_df['bar_code'] == image.bar_code, 'FC_volumes_manual')] = FC_volume_manual
#             volume_df.loc[(volume_df['bar_code'] == image.bar_code, 'TC_volumes_manual')] = TC_volume_manual
#         else:
#             volume_df = volume_df.append({'bar_code': image.bar_code,
#                                       'FC_volumes_manual': FC_volume_manual,
#                                       'FC_volumes_model': FC_volume_model,
#                                       'TC_volumes_manual': TC_volume_manual,
#                                       'TC_volumes_model': TC_volume_model},
#                                      ignore_index=True)
#         t.set_description("FC_manual: {} ,TC_manual: {}".format(FC_volume_manual, TC_volume_manual))
#     volume_df.to_csv(volume_file, index=False)
#
# else:
#     volume_df = pd.DataFrame(columns=['bar_code', 'FC_volumes_manual', 'FC_volumes_model', 'TC_volumes_manual', 'TC_volumes_model'])
#     # get model segmentation volumes
#     for image in tqdm(analysis_images):
#         FC_volume_model = get_seg_volume(image.FC_probmap_file)
#         TC_volume_model = get_seg_volume(image.TC_probmap_file)
#
#         image_records = chart.loc[(chart['src_subject_id'] == image.patient_id) &
#                                      (match_image_date_to_chart_dates(chart['interview_date'], image.study_date))]
#
#         FC_volume_manual = process_score_sets(image_records['wfwvol'].values)
#         TC_volume_manual = process_score_sets(image_records['wmtvcl'].values) + \
#                            process_score_sets(image_records['wltvcl'].values)
#
#         volume_df = volume_df.append({'bar_code': image.bar_code,
#                           'FC_volumes_manual': FC_volume_manual,
#                           'FC_volumes_model': FC_volume_model,
#                           'TC_volumes_manual': TC_volume_manual,
#                           'TC_volumes_model': TC_volume_model},
#                          ignore_index=True)
#         pass
#         # if not math.isnan(FC_volume_manual) and not math.isnan(FC_volume_model):
#         #     FC_volumes_manual.append(FC_volume_manual)
#         #     FC_volumes_model.append(FC_volume_model)
#         #
#         # if not math.isnan(TC_volume_manual) and not math.isnan(TC_volume_model):
#         #     FC_volumes_manual.append(TC_volume_manual)
#         #     FC_volumes_model.append(TC_volume_model)
#     volume_df.to_csv(volume_file, index=False)

pass

# wfwvol volume of cartilage - whole femur (F.VC) [mm^3]
# wftvar variance of cartilage thickness - whole femur (F.ThCtAB) [mm^2]
# wftmax maximum cartilage thickness - whole femur (F.ThCtAB) [mm]
# wftavg mean cartilage thickness - whole femur (F.ThCtAB) [mm
# wfnvol normalized cartilage volume - whole femur (F.VCtAB) [mm]

# wmtvcl volume of cartilage - medial tibia  (MT.VC) [mm^3]
# wltvcl volume of cartilage - lateral tibia (LT.VC) [mm^3]

# d = pd.read_pickle('thickness_results.pkl')
# fc = np.load('thickness_results_femoral_cartilage.npz', allow_pickle=True)['data']
# tc = np.load('thickness_results_tibial_cartilage.npz', allow_pickle=True)['data']


