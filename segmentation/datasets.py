import SimpleITK as sitk
import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

sys.path.append(os.path.realpath(".."))
import utils.transforms as bio_transform

class NiftiDataset(Dataset):
    """
    a dataset class to load medical image data in Nifti format using simpleITK
    """

    def __init__(self, txt_files, data_dir, mode, preload=False, transform=None, to_right=False):
        """
        
        :param txt_files: txt file with lines of "image_file, segmentation_file" or a list of such files
        :param root_dir: the data root dir
        :param transform: transformations on a sample for data augmentation
        """
        self.data_dir = data_dir
        self.mode = mode
        self.preload = preload
        self.transform = transform
        self.to_right = to_right

        if preload:
            print("Preloading data:")
            self.image_list, self.segmentation_list, self.name_list = self.read_image_segmentation(self.data_dir, txt_files)
            print('Preloaded {} training samples'.format(len(self.image_list)))
            if self.to_right:
                l2r = bio_transform.LeftToRight()
                for i in range(len(self.image_list)):
                    if 'LEFT' in self.name_list[i]:
                        self.image_list[i] = l2r.left_to_right_(self.image_list[i])
                        if self.segmentation_list[i]:
                            self.segmentation_list[i] = l2r.left_to_right_(self.segmentation_list[i])
                print("Transform left knees to a right knee orientation")
        else:
            self.image_list, self.segmentation_list, self.name_list = self.read_image_segmentation_list(txt_files, self.data_dir)


        if len(self.image_list) != len(self.segmentation_list):
            raise ValueError("The numbers of images and segmentations are different")

    def __len__(self):
        return self.image_list.__len__()


    def __getitem__(self, id):
        if self.preload:
            image = self.image_list[id]
            segmentation = self.segmentation_list[id]
        else:
            image_file_name = self.image_list[id]
            segmentation_file_name = self.segmentation_list[id]

            # check file existence
            if not os.path.exists(image_file_name):
                print(image_file_name + ' not exist!')
                return
            if not os.path.exists(segmentation_file_name):
                print(segmentation_file_name + ' not exist!')
                return
            image = sitk.ReadImage(image_file_name)
            segmentation = sitk.ReadImage(segmentation_file_name)

        image_name = self.name_list[id]
        sample = {'image': image, 'segmentation': segmentation, 'name': image_name}

        if not self.preload and self.to_right:
            l2r = bio_transform.LeftToRight()
            sample = l2r(sample)

        if self.transform:
            sample = self.transform(sample)

        # sitk_to_tensor = bio_transform.SitkToTensor()
        # tensor_sample = sitk_to_tensor(sample)
        return sample['image'], sample['segmentation'], sample['name']

    @staticmethod
    def read_image_segmentation_list(text_files, data_root=''):
        """
        Read image filename list (and segmentation filename list) from a text file or a series of files
        :param text_files: name(s) of txt files
        :param data_root: root of image data
        :return: image_list: a list of image filenames
                segmentation_list: a list of segmentation filenames
                name_list: a list of scan name (parsed from image filenames)
        """
        image_list = []
        segmentation_list = []
        name_list = []
        if isinstance(text_files, str):
            text_files = [text_files]
        for text_file in text_files:
            with open(text_file) as file:
                for line in file:
                    try:
                        image, seg = line.strip("\n").split(', ')
                    except ValueError:  # Adhoc for test.
                        image = seg = line.strip("\n")
                    # image = image.split(".")[0]+'_corrected.nii.gz'
                    image_list.append(os.path.join(data_root, image))
                    segmentation_list.append(os.path.join(data_root, seg))
                    name_list.append('_'.join(image.split("_")[:-1]))
        return image_list, segmentation_list, name_list

    @staticmethod
    def read_image_segmentation(text_files, data_root=''):
        """
        Similar to read_image_segmentation_list() but instead of returning lists of filenames,
        it returns list of loaded images (simpleITK image instances)
        :param text_files:
        :param data_root:
        :return:
        """

        image_list = []
        segmentation_list = []
        name_list=[]
        if isinstance(text_files, str):
            text_files = [text_files]

        for text_file in text_files:
            with open(text_file) as file:
                for line in file:
                    try:
                        image_file_name, segmentation_file_name = line.strip("\n").split(', ')
                    except ValueError:  # Adhoc for test.
                        image_file_name = segmentation_file_name = line.strip("\n")

                    name_list.append('_'.join(image_file_name.split("_")[:-1]))

                    image_file_name = os.path.join(data_root, image_file_name)
                    segmentation_file_name = os.path.join(data_root, segmentation_file_name)
                    # check file existence
                    if not os.path.exists(image_file_name):
                        print(image_file_name + ' not exist!')
                        continue
                    if not os.path.exists(segmentation_file_name):
                        print(segmentation_file_name + ' not exist!')
                        continue

                    image_list.append(sitk.ReadImage(image_file_name))
                    segmentation_list.append(sitk.ReadImage(segmentation_file_name))

        return image_list, segmentation_list, name_list


class SegDatasetOAIZIB(Dataset):
    """
    a dataset class to load medical image data in Nifti format using simpleITK
    """

    def __init__(self, txt_files, data_dir, mode, preload=False, transform=None, to_right=False):
        """

        :param txt_files: txt file with lines of "image_file, segmentation_file" or a list of such files
        :param root_dir: the data root dir
        :param transform: transformations on a sample for data augmentation
        """
        self.data_dir = data_dir
        self.mode = mode
        self.preload = preload
        self.transform = transform

        if preload:
            print("Preloading data:")
            self.image_list, self.segmentation_list, self.name_list = self.read_image_segmentation(self.data_dir,
                                                                                                   txt_files)

        else:
            self.image_list, self.segmentation_list, self.name_list = self.read_image_segmentation_list(txt_files,
                                                                                                        self.data_dir)

        if len(self.image_list) != len(self.segmentation_list):
            raise ValueError("The numbers of images and segmentations are different")

    def __len__(self):
        return self.image_list.__len__()

    def __getitem__(self, id):
        if self.preload:
            image = self.image_list[id]
            segmentation = self.segmentation_list[id]
        else:
            image_file_name = self.image_list[id]
            segmentation_file_name = self.segmentation_list[id]

            # check file existence
            if not os.path.exists(image_file_name):
                print(image_file_name + ' not exist!')
                return
            if not os.path.exists(segmentation_file_name):
                print(segmentation_file_name + ' not exist!')
                return
            image = sitk.ReadImage(image_file_name)
            segmentation = sitk.ReadImage(segmentation_file_name)

        image_name = self.name_list[id]
        sample = {'image': image, 'segmentation': segmentation, 'name': image_name}

        if self.transform:
            sample = self.transform(sample)

        # sitk_to_tensor = bio_transform.SitkToTensor()
        # tensor_sample = sitk_to_tensor(sample)
        return sample['image'], sample['segmentation'], sample['name']

    @staticmethod
    def read_image_segmentation_list(text_files, data_root=''):
        """
        Read image filename list (and segmentation filename list) from a text file or a series of files
        :param text_files: name(s) of txt files
        :param data_root: root of image data
        :return: image_list: a list of image filenames
                segmentation_list: a list of segmentation filenames
                name_list: a list of scan name (parsed from image filenames)
        """
        image_list = []
        segmentation_list = []
        name_list = []
        if isinstance(text_files, str):
            text_files = [text_files]
        for text_file in text_files:
            with open(text_file) as file:
                for line in file:
                    image_name = line.strip("\n")
                    name_list.append(image_name)
                    image_list.append(os.path.join(data_root, image_name + "_image.nii.gz"))
                    segmentation_list.append(os.path.join(data_root, image_name + "_masks.nii.gz"))

        return image_list, segmentation_list, name_list

    @staticmethod
    def read_image_segmentation(text_files, data_root=''):
        """
        Similar to read_image_segmentation_list() but instead of returning lists of filenames,
        it returns list of loaded images (simpleITK image instances)
        :param text_files:
        :param data_root:
        :return:
        """

        image_list = []
        segmentation_list = []
        name_list = []
        if isinstance(text_files, str):
            text_files = [text_files]

        for text_file in text_files:
            with open(text_file) as file:
                for line in file:
                    image_name = line.strip("\n")
                    name_list.append(image_name)
                    image_file_name = os.path.join(data_root, image_name + "_image.nii.gz")
                    segmentation_file_name = os.path.join(data_root, image_name + "_masks.nii.gz")
                    # check file existence
                    if not os.path.exists(image_file_name):
                        print(image_file_name + ' not exist!')
                        continue
                    if not os.path.exists(segmentation_file_name):
                        print(segmentation_file_name + ' not exist!')
                        continue

                    image_list.append(sitk.ReadImage(image_file_name))
                    segmentation_list.append(sitk.ReadImage(segmentation_file_name))

        return image_list, segmentation_list, name_list