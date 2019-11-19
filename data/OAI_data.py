#!/usr/bin/env python3
"""
Classes for OAI data

Created by zhenlinx on 7/3/18
"""

import os
import sys
from functools import reduce

sys.path.append(os.path.realpath(".."))

from shutil import copyfile
import pandas as pd
import SimpleITK as sitk

# from collections import defaultdict
#
# def nested_dict():
#     return defaultdict(nested_dict)


class OAIData:
    def __init__(self, data_sheet=None, raw_data_root=None):
        self.images = list() # a list of OAIImage objects
        self.visit_description = {-1: 'SCREENING',  # only exists in Xray images
                                  0: 'ENROLLMENT',
                                  12: '12 MONTH',
                                  24: '24 MONTH',
                                  36: '36 MONTH',
                                  48: '48 MONTH',
                                  72: '72 MONTH',
                                  96: '96 MONTH'}
        self.root_path = None  # root path of data and initialized when building patient repositories
        self.patient_set = set()
        if data_sheet != None and raw_data_root != 'None':
            self.load_raw_data_sheet(data_sheet, raw_data_root=raw_data_root)
        else:
            print("WARNING: the OAI data is not correctly initialized since both data_sheet and raw_data_root are required!")

    def load_raw_data_sheet(self, data_sheet, raw_data_root='', proceesed_data_root=''):
        """
        load images info from a data sheet
        raw_data_root is the root path of raw OAI images

        :param data_sheet: data sheet whose entries(rows) are info of each image
        The columns of sheet are:
            Folder,ParticipantID,StudyDate,Barcode,StudyDescription,SeriesDescription
        :param raw_data_root: root path the raw image data
        :return:
        """
        self.df = pd.read_csv(data_sheet)

        for df_line in self.df.itertuples(index=False):
            image = OAIImage(df_line, raw_data_root)
            self.images.append(image)
            self.patient_set.add(image.patient_id)

    def set_processed_data_paths(self, proceesed_data_root='',task_name=None):
        """Set the root folder where the processed data are saved"""
        self.root_path = proceesed_data_root
        for image in self.images:
            image.set_processed_data_paths(proceesed_data_root,task_name)

    def set_processed_data_paths_without_creating_image_directories(self, proceesed_data_root='',task_name=None):
        """Set the root folder where the processed data are saved"""
        self.root_path = proceesed_data_root
        for image in self.images:
            image.set_processed_data_paths_without_creating_directories(proceesed_data_root,task_name)


    def build_repositories(self):
        """
        Build nested repositories for patients to save processed OAI data
        e.g. intermediate results and add preprocessed images
        The structure of repositories are :
        -patient_ids
            -modalities (e.g. MR_SAG_3D_DESS ...)
                -parts (e.g. 'LEFT KNEE', 'RIGHT KNEE' ...)
                    -visits (e.g. 'ENROLLMENT', '12 MONTH', ...)
                        -image (e.g. 'image_normalized.nii.gz')
                        -segmentation folder (named by the CNN segmentation checkpoint name)
                            - segmentation results: cartilage_FC_TC_mask,
                            - meshes of cartilages:
        """

        for image in self.images:
            if not os.path.exists(image.folder):
                os.makedirs(image.folder, exist_ok=True)

    # def preprocess_images(self, old_normalized_image_folder=None, overwrite=False):
    #     """preprocess raw images and save them into corresponding folders"""
    #     images = self.get_images()
    #     for image in images:
    #         image.preprocessed_image_file = os.path.join(image.folder, "image_normalized.nii.gz")
    #
    #         os.makedirs(image.folder, exist_ok=True)
    #         if overwrite or (not os.path.isfile(image.preprocessed_image_file)) :
    #             # if there were processed image saved in a single folder, copy them into the folder
    #             if old_normalized_image_folder:
    #                 old_normalized_image_file_name = "{}_{}_{}_{}_image.nii.gz".format(image.patient_id,
    #                                                                             image.study_date,
    #                                                                             image.series_description,
    #                                                                             image.bar_code)
    #                 old_normalized_image_file_path = os.path.join(old_normalized_image_folder, old_normalized_image_file_name)
    #                 print("Copying {} to {}".format(old_normalized_image_file_path, image.preprocessed_image_file))
    #                 copyfile(old_normalized_image_file_path, image.preprocessed_image_file)
    #             else:
    #                 # TODO: load images from raw folders and do preprocessing
    #                 # preprocess(visit['raw_folder'], tmp_new_image_path)
    #                 pass


    def clean_data(self):
        # clean dataset by checking the consistency of resolution
        """
        abnormal example list:
        !! image size not matched , img:9901199_20090422_SAG_3D_DESS_RIGHT_12800503_image.nii.gz sz:(160, 384, 352)
        !! image size not matched , img:9052335_20090126_SAG_3D_DESS_RIGHT_12766414_image.nii.gz sz:(176, 384, 384)
        !! image size not matched , img:9163391_20110808_SAG_3D_DESS_LEFT_16613250603_image.nii.gz sz:(159, 384, 384)
        !! image size not matched , img:9712762_20090420_SAG_3D_DESS_RIGHT_12583306_image.nii.gz sz:(160, 384, 352)
        !! image size not matched , img:9388265_20040405_SAG_3D_DESS_LEFT_10016906_image.nii.gz sz:(176, 384, 384)
        !! image size not matched , img:9388265_20040405_SAG_3D_DESS_LEFT_10016903_image.nii.gz sz:(176, 384, 384)
        !! image size not matched , img:9938453_20071130_SAG_3D_DESS_RIGHT_12140103_image.nii.gz sz:(159, 384, 384)
        !! image size not matched , img:9452305_20070228_SAG_3D_DESS_RIGHT_11633112_image.nii.gz sz:(109, 384, 384)
        !! image size not matched , img:9219500_20080326_SAG_3D_DESS_RIGHT_12266509_image.nii.gz sz:(8, 384, 384)
        !! image size not matched , img:9011949_20060118_SAG_3D_DESS_LEFT_10667703_image.nii.gz sz:(156, 384, 384)
        !! image size not matched , img:9885303_20051212_SAG_3D_DESS_LEFT_10624403_image.nii.gz sz:(155, 384, 384)
        !! image size not matched , img:9833782_20090519_SAG_3D_DESS_RIGHT_12802313_image.nii.gz sz:(176, 384, 384)
        !! image size not matched , img:9462278_20050524_SAG_3D_DESS_RIGHT_10546912_image.nii.gz sz:(156, 384, 384)
        !! image size not matched , img:9126260_20060921_SAG_3D_DESS_RIGHT_11309309_image.nii.gz sz:(66, 384, 384)
        !! image size not matched , img:9487462_20081003_SAG_3D_DESS_RIGHT_11495603_image.nii.gz sz:(176, 384, 384)
        !! image size not matched , img:9847480_20081007_SAG_3D_DESS_RIGHT_11508512_image.nii.gz sz:(159, 384, 384)
        !! image size not matched , img:9020714_20101207_SAG_3D_DESS_RIGHT_16613171935_image.nii.gz sz:(118, 384, 384)
        """

        series_to_clean = []
        series = self.get_image_series()
        for single_series in series:
            if not self._check_images_resolution_consistency(list(single_series.values())):
                print("{}_{}_{} series have inconsistent resolutions!".format(single_series[0].get_brief_info[0],
                                                                              single_series[0].get_brief_info[1],
                                                                              single_series[0].get_brief_info[2]))
                for image in single_series.values:
                    self.images.remove(image)



    def _check_images_resolution_consistency(self, image_series):
        """check the images of patients and return if their resolutions are consistent"""
        assert isinstance(image_series, list), "input has to be a list"
        resolutions = set()
        for image in image_series:
            image_path = image.preprocessed_image_path
            tmp_img = sitk.Image()
            if os.path.isdir(image_path):
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(image_path)
                _ = reader.SetFileNames(dicom_names)
                tmp_img = reader.Execute()
            elif os.path.isfile(image_path):
                tmp_img = sitk.ReadImage(tmp_img)
            else:
                ValueError("No such a file or path at {}".format(tmp_img))
            resolutions.add(tmp_img.GetSize())
        if len(resolutions) == 1:
            return True
        elif len(resolutions) > 1:
            return False
        else:
            ValueError("Error in checking resolutions")

    def get_patient_list(self):
        return list(self.patient_set)

    def get_images(self, **kwargs):
        """
        Get a list of images that meets sepecific requirements e.g. patient_id, modality, body part, visit time et.al.
        Optional
        :param patient_id (list of ints): requested patient ids for all patients, e.g. [9010952,...]
        :param modality (list of strings): requested modalities or None for all modalities e.g. ['MR_SAG_3D_DESS']
        :param part (list of strings): requested body parts or None for all parts e.g. ['LEFT KNEE', 'RIGHT_KNEE', ..]
        :param visit_month (list of ints): visiting time (count of months) e.g. [0, 12, 24, ...]
        :return: a list of images
        Warning!: The returned data are the references of data in the object, any change on them will affect the object
        """
        conditions = {key:value for key, value in kwargs.items() if value is not None}

        images = [image for image in self.images if image.filter(conditions)]

        return images

    def get_image_series(self, with_series_description, **kwargs):
        """
        Get a list of image series of specific patients, modalities and body parts of all visiting times.
        Optional:
        :param patient_id (list of ints): requested patient ids for all patients, e.g. [9010952,...]
        :param modality (list of strings): requested modalities or None for all modalities e.g. ['MR_SAG_3D_DESS']
        :param part (list of strings): requested body parts or None for all parts e.g. ['LEFT KNEE', 'RIGHT_KNEE', ..]
        :return:if with_series_description is True: return a list of pairs of series and series_info from following two:
                    images_series (a list of dictionaries): image series which are dictionaries with visiting num as keys
                    image_series_info (a list of tuples): list of (image patient_id, modality, part)
                Otherwise just return image_series

        Warning!!!The returned data are the references of data in the object, any change on them will affect the object
        """
        images_series = []
        image_series_info = []
        for image in self.images:
            if image.filter(kwargs):
                image_info = image.get_brief_info()
                series_info = image_info[:-1]
                visit = image_info[-1]
                try:
                    # if a series containing this image exists
                    images_series[image_series_info.index(series_info)][visit] = image
                except ValueError:
                    images_series.append({visit: image})
                    image_series_info.append(series_info)
        if with_series_description:
            return [(image_series_info[i], images_series[i]) for i in range(len(image_series_info))]
        else:
            return images_series

    def get_image_series_attributes(self, patient_id, modality, part, attribute):
        """
        Get attributes of a series of image of specific patient at all visiting times, with given modalities, parts.
        """
        # TODO: may not needed
        pass
        # image_paths = {}
        # for visit_num in self.images[patient_id][modality][part]:
        #     image_paths[visit_num] = self.images[patient_id][modality][part][visit_num][attribute]
        # return image_paths

    def get_processed_data_frame(self):
        """generate a data frame from processed image info"""
        df_lines = [image.get_dataframe_line() for image in self.images]
        return pd.concat(df_lines).reset_index()

    def save_datasheet(self, file_name):
        """
        save current image infos into .csv datasheet
        :param file_name:
        :return:
        """
        self.get_processed_data_frame().to_csv(file_name)


class OAIImage:

    def __init__(self, df_line=None, raw_root=None):
        # attributes from raw OAI data
        self.raw_folder = None
        self.patient_id = None
        self.study_date = None
        self.bar_code = None
        self.modality= None
        self.part = None
        self.visit_month = None
        self.visit_description = {-1: 'SCREENING',  # only exists in Xray images
                                  0: 'ENROLLMENT',
                                  12: '12_MONTH',
                                  24: '24_MONTH',
                                  36: '36_MONTH',
                                  48: '48_MONTH',
                                  72: '72_MONTH',
                                  96: '96_MONTH'}

        # attributes of analysis file paths
        self.folder = None
        self.preprocessed_image_file = None
        self.FC_probmap_file = None
        self.TC_probmap_file = None
        self.FC_mesh_file = None
        self.TC_mesh_file = None
        self.affine_transform_file = None
        self.bspline_transform_file = None
        self.warped_FC_mesh_file = None
        self.warped_TC_mesh_file = None
        self.inv_transform_to_atlas = None
        self.FC_thickness_mapped_to_atlas_mesh = None
        self.TC_thickness_mapped_to_atlas_mesh = None
        self.FC_2D_thickness_grid = None
        self.TC_2D_thickness_grid = None

        if df_line:
            self.patient_id = df_line.ParticipantID
            self.study_date = df_line.StudyDate
            self.bar_code = df_line.Barcode
            self.modality, self.part, self.visit_month = self.factor_description(df_line.StudyDescription,
                                                                                 df_line.SeriesDescription)
            if raw_root:
                self.raw_folder = os.path.join(raw_root, df_line.Folder)



    @property
    def name(self):
        return "_".join([str(self.patient_id), self.visit_description[self.visit_month], self.part, self.modality])

    def create_output_directory(self,task_name=None):
        task_folder = os.path.join(self.folder, task_name) if task_name else self.folder
        os.makedirs(task_folder, exist_ok=True)

    def set_processed_data_paths_without_creating_directories(self, processed_root,task_name=None):
        """
        According to the root path of processed data, setup the path of all analysis file
        :param processed_root:
        :return:
        """
        if self.folder is None:
            self.folder = os.path.join(processed_root,
                                       str(self.patient_id) if (self.patient_id is not None) else '',
                                       self.modality if (self.modality is not None) else '',
                                       self.part if (self.part is not None) else '',
                                       self.visit_description[self.visit_month] if (self.visit_month is not None ) else '')

        task_folder = os.path.join(self.folder,task_name) if task_name else self.folder

        self.preprocessed_image_file = os.path.join(self.folder, 'image_preprocessed.nii.gz')
        self.FC_probmap_file = os.path.join(self.folder, 'FC_probmap.nii.gz')
        self.TC_probmap_file = os.path.join(self.folder, 'TC_probmap.nii.gz')
        self.FC_mesh_file = os.path.join(task_folder, "FC_mesh_world.ply")
        self.TC_mesh_file = os.path.join(task_folder, "TC_mesh_world.ply")
        self.affine_transform_file = os.path.join(self.folder, "affine_transform_to_atlas.txt")
        self.bspline_transform_file = os.path.join(self.folder, "bspline_control_points_to_atlas.nii.gz")
        self.warped_FC_mesh_file = os.path.join(task_folder, "FC_mesh_world_to_atlas.ply")
        self.warped_TC_mesh_file = os.path.join(task_folder, "TC_mesh_world_to_atlas.ply")
        self.inv_transform_to_atlas = os.path.join(task_folder, "inv_transform_to_atlas.nii.gz")
        self.FC_thickness_mapped_to_atlas_mesh = os.path.join(task_folder, "atlas_FC_mesh_with_thickness.ply")
        self.TC_thickness_mapped_to_atlas_mesh = os.path.join(task_folder, "atlas_TC_mesh_with_thickness.ply")

        # TODO: naming the file of 2d thickness grid
        self.FC_2D_thickness_grid = os.path.join(task_folder, "FC_2d_thickness")
        self.TC_2D_thickness_grid = os.path.join(task_folder, "TC_2d_thickness")


    def set_processed_data_paths(self, processed_root,task_name=None):
        """
        According to the root path of processed data, setup the path of all analysis file
        :param processed_root:
        :return:
        """
        self.set_processed_data_paths_without_creating_directories(processed_root=
                                                                   processed_root, task_name=task_name)

        self.create_output_directory(task_name=task_name)




    def get_dataframe_line(self):
        return pd.DataFrame([self.__dict__])

    def get_all_attributes(self):
        return self.__dict__

    def get_brief_info(self):
        """Get the patient_id, modality, part, and visit_info of a image"""
        return (self.patient_id, self.modality, self.part, self.visit_month)


    def filter(self, kwargs):
        """
        filter image by given allowed attributes value, e.g. image.filter({'patient_id':[9010952]})
        :param kwargs: a dictionary with image attribute names as keys and the allowed values (as list) as dict values.
                    For example, {'patient_id':[9010952, 9010953], 'part':['LEFT KNEE', 'RIGHT KNEE']}
        :return: True if all filter condition meets
        """
        if kwargs:
            return reduce(lambda x, y: x & y, [self.__getattribute__(key) in value for key, value in kwargs.items()],
                          True)
        else:
            return True

    @staticmethod
    def factor_description(study_description, series_description):
        """
        factorize image descriptions in OAI data sheets and return its modality, visit date, and imaging part

        :param study_description:
        :param series_description:
        :return:
        """
        study_factors = study_description.rstrip().replace('Thigh', 'THIGH').split(
            '^')  # clean descriptions capitalization

        part = study_factors[-1]

        if study_factors[2] == 'SCREENING':
            visit_mon = -1
        elif study_factors[2] == 'ENROLLMENT':
            visit_mon = 0
        else:
            visit_mon = int(study_factors[2].split(' ')[0])

        modality = study_factors[1]

        if modality == "MR":
            series_factor = series_description.split('_')
            # add specific modality info
            if len(series_factor) > 1:
                modality = '_'.join([modality] + series_factor[:-1])
            else:
                modality = '_'.join([modality] + series_factor)

            # LEFT or RIGHT in MR images mean for knee only
            if part == "LEFT" or part == "RIGHT":
                part = part + '_'+'KNEE'
        return modality, part, visit_mon

    def __repr__(self):
        return self.name

class OAIPatients:
    """
    interface to visit patient info chart

    useful attributes:
    'P02RACE': race,
        e.g. '1: White or Caucasian', '2: Black or African American',
            '0: Other Non-white', '3: Asian', '.: Missing Form/Incomplete Workbook'
    'P02SEX': sex,
        e.g. '1: Male', '2: Female'
    'V00COHORT': cohort,
        e.g. '1: Progression', '2: Incidence', '3: Non-exposed control group'
    """
    def __init__(self, enrollees_text_file):
        df = pd.read_csv(enrollees_text_file, sep='|')
        self.df = df.set_index('ID')

    def filter_patient(self, **kwargs):
        df_bool = reduce(lambda x, y: x & y, [self.df[key] == value for key, value in kwargs.items()],
               True)
        return set(self.df.index[df_bool])

    def get_all_attributes(self):
        return self.df.keys()

    def get_unique_attribute_values(self, attribute):
        return self.df[attribute].unique()

def oai_data_test():
    """
    Analyze OAI MR images, including
    - Intensity normalization
    - Femoral/Tibial cartilage segmentation using CNNs
    - Generating cartilage surface meshes from segmentation results
    - Computing (per-vertex) thickness of cartilage
    """
    # set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    os.environ["CUDA_CACHE_PATH"] = "/playpen/zhenlinx/.cuda_cache"

    # patient infos
    patients_ASCII_file_path = "./Enrollees.txt"
    # patients_df = pd.read_table(patients_ASCII_file_path, sep='|')
    # patients_df = patients_df.set_index('ID')
    # progression_cohort = list(patients_df.index[patients_df['V00COHORT'] == '1: Progression'])

    oai_patients = OAIPatients(patients_ASCII_file_path)
    progression_cohort = oai_patients.filter_patient(V00COHORT='1: Progression')

    # image_folder_path = "/playpen-raid/zhenlinx/Data/OAI_segmentation/Nifti_6sets_rescaled"
    # OAI_data_sheet = "/playpen-raid/zhenlinx/Data/OAI_segmentation/SEG_3D_DESS_6sets_40test.csv"
    OAI_data_sheet = "./SEG_3D_DESS_all.csv"

    OAI_data = OAIData(OAI_data_sheet, '/playpen-raid/data/OAI')
    # OAI_data.set_processed_data_paths()
    ## build repositories and preprocess the raw image data
    OAI_data.set_processed_data_paths_without_creating_image_directories('/playpen-raid/zhenlinx/Data/OAI')
    images = OAI_data.get_images()
    series = OAI_data.get_image_series()
    df = OAI_data.get_processed_data_frame()


if __name__ == '__main__':
    oai_data_test()
