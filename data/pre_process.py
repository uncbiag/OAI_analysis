import SimpleITK as sitk
import os
import numpy as np
from glob import glob
from multiprocessing import Pool, TimeoutError
from itertools import product
from functools import partial

# a function to get the center coordinate of a 3D bounding box give its starting position and size
def get_box_center(box):
    box = np.array(box)
    starting_pos = box[0:3]
    size = box[3:]

    return (starting_pos+size//2).astype(int)


# a function to get the up and lower bound of a region given its center index and region size
def get_region_bound(center, size):
    lower = center - (size//2)
    up = lower + size
    return lower, up


def flip_left_right(image):
    image_np = sitk.GetArrayViewFromImage(image)
    image_np = np.flip(image_np, 0)
    image_flipped = sitk.GetImageFromArray(image_np)
    image_flipped.CopyInformation(image)
    return image_flipped

# # crop the image given a region, if the region is outside the image boundaries, padding it with zeros.
# def crop(image, start, size):
#     image_size = image.GetSize()
#     crop_size = np.array([size[i] if start[i]+size[i]<= image_size[i] else image_size[i] - start[i] for i in range(3)])
#     print(crop_size)
#     return sitk.RegionOfInterest(image, crop_size, start)


def crop_sample(image, label, roi_size):
    image_size = image.GetSize()

    label_shape_stat = sitk.LabelShapeStatisticsImageFilter()   # filter to analysis the label shape
    label_shape_stat.Execute(label)
    box = np.array(label_shape_stat.GetBoundingBox(1))

    # get the lower bounds and upper bounds coordinates
    low_bound, up_bound = get_region_bound(get_box_center(box), (roi_size * 1.1).astype(int))

    # get the crop size at lower bounds and limit them to be positive
    # then pad the dropped size after cropping
    crop_low = np.array([low_bound[i] if low_bound[i]>0 else 0 for i in range(3)], dtype=int)
    padding_low = np.array([-low_bound[i] if low_bound[i]<0 else 0 for i in range(3)], dtype=int)

    # get the crop size at upper bound and limit them to be smaller than image size,
    # then pad the dropped size after cropping
    up_bound_diff = image_size - up_bound
    crop_up = np.array([up_bound_diff[i] if up_bound_diff[i]>0 else 0 for i in range(3)], dtype=int)
    padding_up = np.array([-up_bound_diff[i] if up_bound_diff[i]<0 else 0 for i in range(3)], dtype=int)

    # crop and pad
    valid_size = image_size - crop_up - crop_low  # size of valid cropped region
    print("valid cropped size: {}".format(valid_size))
    image_crop = sitk.ConstantPad(sitk.Crop(image, crop_low.tolist(), crop_up.tolist()), padding_low.tolist(), padding_up.tolist())
    label_crop = sitk.ConstantPad(sitk.Crop(label, crop_low.tolist(), crop_up.tolist()), padding_low.tolist(), padding_up.tolist())

    return image_crop, label_crop


def image_normalize(image, window_min_perc, window_max_perc, output_min, output_max):
    window_rescale = sitk.IntensityWindowingImageFilter()
    image_array = sitk.GetArrayFromImage(image)
    window_min = np.percentile(image_array, window_min_perc)
    window_max = np.percentile(image_array, window_max_perc)
    return window_rescale.Execute(image, window_min, window_max, output_min, output_max)

def label2image(label_array, source_image):
    label_image = sitk.GetImageFromArray(label_array)
    label_image.CopyInformation(source_image)
    return label_image


def pre_process(scan_dict, save_dir, if_corrected=True, if_crop=True, if_normalize=True, if_overwrite=False, flip_side=None):
    """
    Preprocess images given the absolute path to them.
    -Pre-processed images are saved with under the same root dir of the data folder with folder name + '_{$ops}'
    :param scan_dict: A dictionary that contains the following entries:
                *image_list: A list of absolute path of images(Nifti file paths, or a tuple (dicom folder, scan description))
    to be preprocessed
                *seg_list(optional): A list of segmentation mask, used when cropping ROI region is chosen
                *name_list(optional): REQUIRED when images are DICOM series.
                    If inputs are Nifti formats, names can be extracted from image file name.
    :param save_dir: directory to save the processed images
    :param if_corrected: if do bias field correction
    :param if_crop: if crop the foreground region based on segmentation mask
    :param if_normalize: if normalize image intensity into [0.1]
    :param if_overwrite: if overwrite existing image files
    :param flip_side: if flip all image to a constant side e.g. 'left' means flip all left image to right
    :return: None
    """
    # get entries from dictionary
    if 'image_list' in scan_dict.keys():
        image_list = scan_dict['image_list']
        if 'seg_list' in scan_dict.keys():
            seg_list = scan_dict['seg_list']
        else:
            if if_crop:
                ValueError('Cannot find key \'seg_list\': segmentations are required to do cropping')
        if 'name_list' in scan_dict.keys():
            name_list = scan_dict['name_list']

    else:
        ValueError("Cannot find key \'image_list\': a list of image paths is required")

    for ind, image_path in enumerate(image_list):

        print("[{}]Processing {}".format(ind, image_path))
        input_dir = os.path.dirname(image_path)

        # TODO: this does not cover when image path are folders of DICOM deries
        if 'name_list' in locals():
            scan_name = name_list[ind]
        else:
            scan_name = "_".join(os.path.basename(image_path).split(".")[0].split("_")[:-1])

        if (not if_overwrite) and os.path.isfile(os.path.join(save_dir, scan_name + '_image.nii.gz')):
            if if_crop:
                if os.path.isfile(os.path.join(save_dir, scan_name + '_label_all.nii.gz')):
                    print("{} exists".format(scan_name + '_label_all.nii.gz'))
                    continue
                else:
                    pass
            else:
                print("{} exists".format(scan_name + '_image.nii.gz'))
                continue
        else:
            pass

        # read image and segmentation file
        if os.path.isfile(image_path):
            # read NIFTI image
            try:
                image = sitk.ReadImage(image_path)
            except:
                print("Can not find a NIFTI file {}".format(image_path))

            # # get image name
            # if 'name_list' in locals():
            #     scan_name = name_list[ind]
            # else:
            #     scan_name = os.path.basename(image_path).split(".")[0]

        elif os.path.isdir(image_path):
            # scan_name = name_list[ind]
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(image_path)

            print("Read {} DICOM slices in {}".format(len(dicom_names), image_path))
            # reader.GetGDCMSeriesIDs
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
        else:
            ValueError("{} is neither a valid NIFTI file nor a folder containing DICOM series".format(image_path))

        # #code for calculate the ROI size
        # label_shape_stat = sitk.LabelShapeStatisticsImageFilter()   # filter to analysis the label shape
        # label_shape_stat.Execute(label)
        # if box == []:
        #     box = np.array(label_shape_stat.GetBoundingBox(1), ndmin=2)
        # else:
        #     box = np.concatenate((box, np.array(label_shape_stat.GetBoundingBox(1),ndmin=2)),axis=0)

        image = sitk.Cast(image, sitk.sitkFloat32)

        if 'seg_list' in locals():
            # label_file = os.path.join(input_dir, '_'.join(os.path.basename(image_file).split("_")[:-1]+['label', 'all']) + ".nii.gz")
            label = sitk.ReadImage(seg_list[ind])

        # bias field correction
        if if_corrected:
            print("Bias Correcting " + scan_name)
            image_corrected_file_path = os.path.join(input_dir, scan_name+'_all_corrected.nii.gz')
            if not if_overwrite and os.path.isfile(image_corrected_file_path):
                print("Bias corrected file found!")
                image_corrected = sitk.ReadImage(image_corrected_file_path)

            else:
                # label_comb = sitk.Threshold(label, lower=1, upper=2, outsideValue=0)  # use the label as correcting mask
                all_mask = label2image(np.ones(sitk.GetArrayFromImage(image).shape).astype(int), image) # if want to use all voxels
                image = sitk.Add(image, 1)
                image = sitk.N4BiasFieldCorrection(image, maskImage=all_mask)
                image = sitk.Subtract(image, 1)
                print("Saving corrected: " + scan_name)
                sitk.WriteImage(image, image_corrected_file_path)

        # rescale the intensity
        if if_normalize:
            print("Normalizing: {}".format(scan_name))
            image = image_normalize(image, 0.1, 99.9, 0, 1)

        # crop the ROI
        if if_crop:
            print("Cropping: " + scan_name)
            roi_size = np.array([228, 167, 139])  # this size is pre-calculated
            image, label = crop_sample(image, label, roi_size)  # crop the ROI

        # reset original and orientation
        reset_sitk_image_coordinates(image, [0, 0, 0], [0, 0, -1, 1, 0, 0, 0, -1, 0])

        # if flip the left and right:
        if flip_side and (flip_side in scan_name):
            print("flipping {} to {} image".format(scan_name, flip_side))
            image = flip_left_right(image)
            if 'seg_list' in locals():
                label = flip_left_right(label)

        # save the images
        print("Saving:{} at {}".format(scan_name, save_dir))
        sitk.WriteImage(image, os.path.join(save_dir, scan_name + '_image.nii.gz'))
        if 'seg_list' in locals():
            sitk.WriteImage(label, os.path.join(save_dir, scan_name + '_label_all.nii.gz'))


def pre_process_parallel(image_list, save_dir, seg_files=None, name_list=None, n_workers=1, if_corrected=False, if_normalize=True, if_crop=False,
                         if_overwrite=False, flip_side=None):
    """
    run pre_process() in parallel.
    :param image_list: A list of image path to be process
    :param save_dir: directory to save the processed images
    :param if_corrected: if do bias field correction
    :param if_crop: if crop the foreground region based on segmentation mask
    :param if_normalize: if normalize image intensity into [0.1]
    :param if_overwrite: if overwrite existing image files
    :param flip_side: if flip all image to a constant side e.g. 'left' means flip all left image to right
    :return: None
    """

    # np.random.shuffle(image_files) # cannot remember why the list was shuffled
    image_list_partitions = np.array_split(image_list, n_workers)
    scan_dict_partitions = [{'image_list': image_list_partitions[i]} for i in range(n_workers)]
    if seg_files:
        seg_file_partitions = np.array_split(seg_files, n_workers)
        for i in range(n_workers):
            scan_dict_partitions[i]['seg_list'] = seg_file_partitions[i]

    if name_list:
        name_list_partitions = np.array_split(name_list, n_workers)
        for i in range(n_workers):
            scan_dict_partitions[i]['name_list'] = name_list_partitions[i]

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with Pool(processes=n_workers) as pool:
        res = pool.map(partial(pre_process, save_dir=save_dir, if_corrected=if_corrected, if_crop=if_crop,
                               if_normalize=if_normalize, if_overwrite=if_overwrite, flip_side=flip_side),
                       scan_dict_partitions)

    # box = []  # ROI regions of each image
    # print(np.max(box,axis=0)[3:])  # get the max bounding box for ROI which gave [228 150 135]X [228 167 139]

def reset_image_coordinates(image_file, origin=None, orientation=None):
    """
    Reset the origin and orientation of a given image file
    :param image_file:
    :param origin: [x,y,z]
    :param orientation: flat vector of orientation matrix in row major
            [o11, o12, 13, o21, o22, o23, o31, o32, o33]
    :return:
    """
    sitkImg = sitk.ReadImage(image_file)
    reset_sitk_image_coordinates(sitkImg, origin, orientation)
    sitk.WriteImage(sitkImg, image_file)


def reset_sitk_image_coordinates(sitkimage, origin=None, orientation=None):
    """
    Reset the origin and orientation of a given sitk image object
    :param image_file:
    :param origin:
    :param orientation:
    :return:
    """
    if origin:
        sitkimage.SetOrigin(origin)
    if orientation:
        sitkimage.SetDirection(orientation)

def main():

    nifti_dir = os.path.realpath("../data/Nifti")  # repository raw nifti file
    nifti_dir = "/playpen-raid/zhenlinx/Data/OAI_segmentation/Nifti_rescaled"  # repository raw nifti file
    image_list = sorted(glob(os.path.join(nifti_dir, "*_image.nii.gz")))  # get image files
    seg_list = sorted(glob(os.path.join(nifti_dir, "*_label_all.nii.gz")))  # get image files

    number_of_workers = 1
    if_corrected = False  # if do bias field correction
    if_normalize = False
    if_crop = False
    flip_side = "RIGHT"
    all_sides = ["RIGHT", "LEFT"]
    save_dir = nifti_dir + '{}{}{}_{}'.format("_corrected" if if_corrected else "",
                                           "_rescaled" if if_normalize else "",
                                           "_cropped" if if_crop else "",
                                              list(set(all_sides) - set([flip_side]))[0] if flip_side else "")

    pre_process_parallel(image_list, save_dir, seg_files=seg_list, n_workers=number_of_workers, if_corrected=if_corrected, if_crop=if_crop,
                               if_normalize=if_normalize, flip_side=flip_side)

if __name__ == '__main__':
    main()
