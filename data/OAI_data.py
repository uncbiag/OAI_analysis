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
    def __init__(self, data_sheet=None, raw_data_root=''):
        self.images = list()
        self.visit_description = {-1: 'SCREENING',  # only exists in Xray images
                                  0: 'ENROLLMENT',
                                  12: '12 MONTH',
                                  24: '24 MONTH',
                                  36: '36 MONTH',
                                  48: '48 MONTH',
                                  72: '72 MONTH'}
        self.root_path = None  # root path of data and initialized when building patient repositories
        self.patient_set = set()
        if data_sheet and raw_data_root:
            self.load_raw_data_sheet(data_sheet, raw_data_root=raw_data_root)

    def load_raw_data_sheet(self, data_sheet, raw_data_root=''):
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
            self.images.append(OAIImage(df_line, raw_data_root))
        pass

    def build_repositories(self, processed_root_path):
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
        self.root_path = processed_root_path
        for image in self.images:
            image.folder = os.path.join(processed_root_path, str(image.patient_id), image.modality, image.part, self.visit_description[image.visit_month])
            if not os.path.exists(image.folder):
                os.makedirs(image.folder, exist_ok=True)

    def preprocess_images(self, old_normalized_image_folder=None, overwrite=False):
        """preprocess raw images and save them into corresponding folders"""
        images = self.get_images()
        for image in images:
            image.normalized_image_path = os.path.join(image.folder, "image_normalized.nii.gz")

            os.makedirs(image.folder, exist_ok=True)
            if overwrite or (not os.path.isfile(image.normalized_image_path)) :
                # if there were processed image saved in a single folder, copy them into the folder
                if old_normalized_image_folder:
                    old_normalized_image_file_name = "{}_{}_{}_{}_image.nii.gz".format(image.patient_id,
                                                                                image.study_date,
                                                                                image.series_description,
                                                                                image.bar_code)
                    old_normalized_image_file_path = os.path.join(old_normalized_image_folder, old_normalized_image_file_name)
                    print("Copying {} to {}".format(old_normalized_image_file_path, image.normalized_image_path))
                    copyfile(old_normalized_image_file_path, image.normalized_image_path)
                else:
                    # TODO: load images from raw folders and do preprocessing
                    # preprocess(visit['raw_folder'], tmp_new_image_path)
                    pass


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
            image_path = image.normalized_image_path
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

        images = [image for image in self.images if image.filter(kwargs)]

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

    @staticmethod
    def factor_description(study_description: str, series_description: str):
        """
        factorize image descriptions in OAI data sheets and return its modality, visit months, and imaging part
        :param study_description:
        :param series_description:
        :return:
        """
        study_factors = study_description.rstrip().replace('Thigh', 'THIGH').split(
            '^')  # clean descriptions capitalization

        part = study_factors[-1]

        # visit_counts = study_factors[2]
        if study_factors[2] == 'SCREENING':
            visit_counts = -1
        elif study_factors[2] == 'ENROLLMENT':
            visit_counts = 0
        else:
            visit_counts = int(study_factors[2].split(' ')[0])

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
                part = part + ' KNEE'
        return [modality, part, visit_counts]

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

        # attributes in OAI analysis
        # self.folder = None
        # self.normalized_image_path = None
        # self.cartilage_segmentation_path = None
        # self.FC_mesh_path = None
        # self.TC_mesh_path = None


        if df_line:
            self.patient_id = df_line.ParticipantID
            self.study_date = df_line.StudyDate
            self.bar_code = df_line.Barcode
            self.modality, self.part, self.visit_month = self.factor_description(df_line.StudyDescription,
                                                                                 df_line.SeriesDescription)

            if raw_root:
                self.raw_folder = os.path.join(raw_root, df_line.Folder)

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
                part = part + ' KNEE'
        return modality, part, visit_mon


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

    image_folder_path = "/playpen-raid/zhenlinx/Data/OAI_segmentation/Nifti_6sets_rescaled"
    OAI_data_sheet = "/playpen-raid/zhenlinx/Data/OAI_segmentation/SEG_3D_DESS_6sets_40test.csv"
    # OAI_data_sheet = "/playpen-raid/zhenlinx/Data/OAI_segmentation/SEG_3D_DESS_6sets.csv"
    OAI_data = OAIData(OAI_data_sheet, '/playpen-raid/data/OAI')
    ## build repositories and preprocess the raw image data
    OAI_data.build_repositories('/playpen-raid/zhenlinx/Data/OAI')
    images = OAI_data.get_images()
    series = OAI_data.get_image_series()
    df = OAI_data.get_processed_data_frame()
    # OAI_data.preprocess_images('/playpen-raid/zhenlinx/Data/OAI', old_image_folder=image_folder_path, overwrite=False)

    # segment images and save masks and probability map
    # checkpoint_path = '../unet_3d/ckpoints/MICCAI_rebuttal/Cascaded_1_AC_residual-1-s1_end2end_multi-out_UNet_bias_' \
    #                   'Nifti_rescaled_train1_patch_128_128_32_batch_4_sample_0.01-0.02_cross_entropy_lr_0.0005_' \
    #                   'scheduler_multiStep_04282018_011610/model_best.pth.tar'
    # cascaded_segmenter = CascadedSegmenter(checkpoint_path)
    # OAI_data.segment_all_images(cascaded_segmenter, overwrite=False)

    # # extract cartilage surface and compute thickness on them.
    # OAI_data.extract_surface_and_compute_thickness(cascaded_segmenter.name, overwrite=True)



if __name__ == '__main__':
    oai_data_test()
