#!/usr/bin/env python
"""
Make datasheets for images and their meta data
Created by zhenlinxu on 07/08/2019
"""
import pandas as pd
import os
import sys

def make_sheet():
    OAI_path = "/playpen-raid/data/OAI/"
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

    # get SAG_3D_DESS knee MRI scan from each file
    SAG_3D_DESS = [None] * 8
    for j in range(8):
        SAG_3D_DESS[j] = set_list[j][set_list[j]['SeriesDescription']
            .str.contains('SAG_3D_DESS')]

        counts = SAG_3D_DESS[j]['ParticipantID'].value_counts()
        print("{} total: {}".format(j, str(SAG_3D_DESS[j].ParticipantID.unique().size)))

    # get the 96month data
    image03file = os.path.join(OAI_path, 'OAINonImaging', 'image03.txt')
    df_96m = get_96mon_sheet(image03file)
    print("{} total: {}".format(8, str(df_96m.ParticipantID.unique().size)))

    # we do not use 18month and 30month data since they are small investigation
    SAG_3D_DESS_all = pd.concat(SAG_3D_DESS[:6] + [df_96m])
    SAG_3D_DESS_all = SAG_3D_DESS_all.sort_values(by=['ParticipantID', 'SeriesDescription', 'StudyDate'],
                                                  ascending=True).reset_index(drop=True)
    print('Total {} MR(SAG_3D_DESS) knee images'.format(len(SAG_3D_DESS_all)))

    # save datasheet
    SAG_3D_DESS_all.to_csv(os.path.join(save_path, "SEG_3D_DESS_all.csv"))

    """
    ================
    Rest of the script is making datasheet for knees with 7 visits
    ================
    """

    # get seperate lists for left and right knees
    SAG_3D_DESS_LEFT = SAG_3D_DESS_all[SAG_3D_DESS_all['SeriesDescription'].str.contains('LEFT')]
    SAG_3D_DESS_RIGHT = SAG_3D_DESS_all[SAG_3D_DESS_all['SeriesDescription'].str.contains('RIGHT')]

    # get patients with exactly 6 sets for right knee
    right_counts = SAG_3D_DESS_RIGHT['ParticipantID'].value_counts()
    right_list_7 = right_counts[right_counts == 7]

    # get patients with exactly 6 sets for left knee
    left_counts = SAG_3D_DESS_LEFT['ParticipantID'].value_counts()
    left_list_7 = left_counts[left_counts == 7]

    # get image files for patients with 7 scans (seperate list for right and left knee)
    SAG_3D_DESS_RIGHT_7 = SAG_3D_DESS_RIGHT[SAG_3D_DESS_RIGHT['ParticipantID'].isin(right_list_7.index)].sort_values(
        by=['ParticipantID', 'SeriesDescription', 'StudyDate'])
    print('Total {} MR(SAG_3D_DESS) Right knee images with 7 visits (0-96month) from {} patients'.format(
        len(SAG_3D_DESS_RIGHT_7), len(right_list_7)))

    SAG_3D_DESS_LEFT_7 = SAG_3D_DESS_LEFT[SAG_3D_DESS_LEFT['ParticipantID'].isin(left_list_7.index)].sort_values(
        by=['ParticipantID', 'SeriesDescription', 'StudyDate'])
    print('Total {} MR(SAG_3D_DESS) Left knee images with 7 visits (0-96month) from {} patients'.format(
        len(SAG_3D_DESS_LEFT_7), len(left_list_7)))

    # combination of left and right list
    list_7visits = pd.concat([SAG_3D_DESS_RIGHT_7, SAG_3D_DESS_LEFT_7])
    print('Total {} MR(SAG_3D_DESS) Right/Left knee images with 7 visits (0-96month)'.format(len(list_7visits)))
    list_7visits = list_7visits.sort_values(by=['ParticipantID', 'SeriesDescription', 'StudyDate'],
                                            ascending=True).reset_index(drop=True)

    list_7visits.to_csv(os.path.join(save_path, "SEG_3D_DESS_7visits.csv"))

def get_96mon_sheet(image03file):
    """
    In the new OAI data, meta-data is not provided seperated for each visit time.
    All image info are stored in image03.txt is the OAINonImaging collection

    The function get dataframe of 96month SAG_3D_DESS from image03.txt
    :return:
    """

    df = pd.read_csv(image03file, sep='\t')

    # get entries with SAG_3D_DESS imaging
    df = df[df['comments_misc'].str.contains('96') & df['image_description'].str.contains('SAG_3D_DESS')]
    # translate to old format
    new_df = pd.DataFrame()

    new_df['Folder'] = [os.path.join(*(v.split('/')[5:-1] + [v.split('/')[-1].split('.')[0]])) for v in
                        df['image_file'][1:]]
    new_df['ParticipantID'] = list(df['src_subject_id'])[1:]
    new_df['StudyDate'] = [''.join([v.split('/')[2], v.split('/')[0], v.split('/')[1]]) for v in
                           df['interview_date'][1:]]
    new_df['Barcode'] = list(df['image03_id'])[1:]
    new_df['StudyDescription'] = ['^'.join(v.split(' ')[:2] + [' '.join(v.split(' ')[2:4])] + v.split(' ')[4:]) for v in
                                  df['comments_misc'][1:]]
    new_df['SeriesDescription'] = list(df['image_description'])[1:]

    return new_df.reset_index(drop=True)

if __name__ == '__main__':
    make_sheet()

