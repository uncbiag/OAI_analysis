#!/usr/bin/env python
"""
Make datasheets for images and their meta data
Created by zhenlinxu on 07/08/2019
"""
import pandas as pd
import os
import sys

def main():
    OAI_path = "/playpen/data/OAI/"
    save_path = '.'
    # read csv files
    set_list = [None] * 8
    set_list[0] = pd.read_csv(os.path.join(OAI_path, 'contents.0E1.csv'))  # baseline 4796
    set_list[1] = pd.read_csv(os.path.join(OAI_path, 'contents.1E1.csv'))  # 12 month 4313
    set_list[6] = pd.read_csv(os.path.join(OAI_path, 'contents.2D2.csv'))  # 18 month 287
    set_list[2] = pd.read_csv(os.path.join(OAI_path, 'contents.3E1.csv'))  # 24 month 4081
    set_list[7] = pd.read_csv(os.path.join(OAI_path, 'contents.4G1.csv'))  # 30 month 483
    set_list[3] = pd.read_csv(os.path.join(OAI_path, 'contents.5E1.csv'))  # 36 month 3950
    set_list[4] = pd.read_csv(os.path.join(OAI_path, 'contents.6E1.csv'))  # 48 month 3803
    set_list[5] = pd.read_csv(os.path.join(OAI_path, 'contents.8E1.csv'))  # 72 month 3142
    list_all = pd.concat(set_list[:6])

    # get SAG_3D_DESS knee MRI scan from each file
    SAG_3D_DESS = [None] * 8
    for j in range(8):
        SAG_3D_DESS[j] = set_list[j][set_list[j]['SeriesDescription']
            .str.contains('SAG_3D_DESS')]

        counts = SAG_3D_DESS[j]['ParticipantID'].value_counts()
        print("{} total: {}".format(j, str(SAG_3D_DESS[j].ParticipantID.unique().size)))

    # we do not use 18month and 30month data since they are small investigation
    SAG_3D_DESS_all = pd.concat(SAG_3D_DESS[:6])
    SAG_3D_DESS_all = SAG_3D_DESS_all.sort_values(by=['ParticipantID', 'SeriesDescription', 'StudyDate'],
                                                  ascending=True).reset_index(drop=True)
    print('Total {} MR(SAG_3D_DESS) knee images'.format(len(SAG_3D_DESS_all)))

    # save datasheet
    SAG_3D_DESS_all.to_csv(os.path.join(save_path, "SEG_3D_DESS_all.csv"))

    """
    ================
    Rest of the script is making datasheet for knees with 6 visits
    ================
    """

    # get seperate lists for left and right knees
    SAG_3D_DESS_LEFT = SAG_3D_DESS_all[SAG_3D_DESS_all['SeriesDescription'].str.contains('LEFT')]
    SAG_3D_DESS_RIGHT = SAG_3D_DESS_all[SAG_3D_DESS_all['SeriesDescription'].str.contains('RIGHT')]

    # get patients with exactly 6 sets for right knee
    right_counts = SAG_3D_DESS_RIGHT['ParticipantID'].value_counts()
    right_list_6 = right_counts[right_counts == 6]

    # get patients with exactly 6 sets for left knee
    left_counts = SAG_3D_DESS_LEFT['ParticipantID'].value_counts()
    left_list_6 = left_counts[left_counts == 6]

    # get image files for patients with 6 scans (seperate list for right and left knee)
    SAG_3D_DESS_RIGHT_6 = SAG_3D_DESS_RIGHT[SAG_3D_DESS_RIGHT['ParticipantID'].isin(right_list_6.index)].sort_values(
        by=['ParticipantID', 'SeriesDescription', 'StudyDate'])
    print('Total {} MR(SAG_3D_DESS) Right knee images with 6 visits (0-72month) from {} patients'.format(
        len(SAG_3D_DESS_RIGHT_6), len(right_list_6)))

    SAG_3D_DESS_LEFT_6 = SAG_3D_DESS_LEFT[SAG_3D_DESS_LEFT['ParticipantID'].isin(left_list_6.index)].sort_values(
        by=['ParticipantID', 'SeriesDescription', 'StudyDate'])
    print('Total {} MR(SAG_3D_DESS) Left knee images with 6 visits (0-72month) from {} patients'.format(
        len(SAG_3D_DESS_LEFT_6), len(left_list_6)))

    # combination of left and right list
    list_6visits = pd.concat([SAG_3D_DESS_RIGHT_6, SAG_3D_DESS_LEFT_6])
    print('Total {} MR(SAG_3D_DESS) Right/Left knee images with 6 visits (0-72month)'.format(len(list_6visits)))
    list_6visits = list_6visits.sort_values(by=['ParticipantID', 'SeriesDescription', 'StudyDate'],
                                            ascending=True).reset_index(drop=True)

    # list_6visits.to_csv(os.path.join(save_path, "SEG_3D_DESS_6visits.csv"))


if __name__ == '__main__':
    main()

