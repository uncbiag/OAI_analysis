#!/usr/bin/env python
"""
This script test the results of thickness computation by comparing them with manual measurements

Created by zhenlinx on 9/17/19
"""

import pandas
import numpy as np


# wfwvol volume of cartilage - whole femur (F.VC) [mm^3]
# wftvar variance of cartilage thickness - whole femur (F.ThCtAB) [mm^2]
# wftmax maximum cartilage thickness - whole femur (F.ThCtAB) [mm]
# wftavg mean cartilage thickness - whole femur (F.ThCtAB) [mm
# wfnvol normalized cartilage volume - whole femur (F.VCtAB) [mm]

# wmtvcl volume of cartilage - medial tibia  (MT.VC) [mm^3]
# wltvcl volume of cartilage - lateral tibia (LT.VC) [mm^3]

# d = pandas.read_pickle('thickness_results.pkl')
# fc = np.load('thickness_results_femoral_cartilage.npz', allow_pickle=True)['data']
# tc = np.load('thickness_results_tibial_cartilage.npz', allow_pickle=True)['data']
# file = '/playpen-raid/zhenlinx/Data/OAI_raw/OAINonImaging/kmriqcart01.txt'
# oai_qcart = pandas.read_csv(file, sep='\t')

# oai_qcart['wfwvol'].convert_objects(convert_numeric=True).values[1:]
# oai_qcart['wmtvcl'].convert_objects(convert_numeric=True).values[1:]
# oai_qcart['wltvcl'].convert_objects(convert_numeric=True).values[1:]
import SimpleITK as sitk
prob  = sitk.ReadImage('/data/raid1/oai_analysis_results/9000099/MR_SAG_3D_DESS/LEFT_KNEE/72_MONTH/TC_probmap.nii.gz')
mask = prob > 0.5
sitk.WriteImage(mask, '/data/raid1/oai_analysis_results/9000099/MR_SAG_3D_DESS/LEFT_KNEE/72_MONTH/TC_mask.nii.gz')

pass