#!/usr/bin/env python3


import os
import sys
import argparse

sys.path.append(os.path.realpath(".."))
# import datasets as data3d

from torchvision import transforms
import misc.transforms as bio_transform
import misc.datasets as data3d
from segmentation.networks import *
from utils.tools import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',
                        default='ckpoints/'
                                'UNet_light2_Nifti_rescaled_patch_248_164_148_batch_1_sample_0.0-0.0_focal_loss_lr_0.0005_scheduler_plateau_01272018_033608/'
                                'model_best.pth.tar',
                        type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--results-dir', default='results', type=str, metavar='PATH',
                        help='path to tensorboard log file (default: none)')
    parser.add_argument('--GPU', default=1, type=int,
                        help='index of used GPU')
    args = parser.parse_args()

    n_classes = 3
    n_channels = 1

    model_name = args.resume.split('/')[1]
    params = model_name.split('_')

    # overlap_size = (0, 0, 0)
    overlap_size = (16, 16, 8)
    test_batch_size = 2  # batch size used for prediction
    testing_list_file = os.path.realpath("../data/validation.txt")  # text file contains image names to be segmented
    data_dir = os.path.realpath("../data/Nifti_rescaled")  # image files path

    patch_size = (int(params[params.index('patch') + 1]), int(params[params.index('patch') + 2]),
                  int(params[params.index('patch') + 3]))  # size of 3d training patches cropped from the original image

    padding_mode = 'reflect'
    if_vote = False
    test_config = "{}_{}{}".format(overlap_size[0], padding_mode, '_vote' if if_vote else '')
    print("Test config: " + test_config)
    partition = bio_transform.Partition(patch_size, overlap_size, padding_mode=padding_mode, mode='pred')
    test_transforms = [partition]

    # set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    # load training data and validation data
    print("load data from {}".format(data_dir))

    test_transform = transforms.Compose(test_transforms)

    testing_data = data3d.NiftiDataset(testing_list_file, data_dir, "training", transform=test_transform)

    # build unet
    model = UNet_light2(in_channel=n_channels, n_classes=n_classes, BN='BN' in params,
                        bias='bias' in params)
    model.cuda()

    # resume checkpoint or initialize
    training_epochs = 0
    if args.resume is None:
        raise ValueError("The path of trained model is needed.")
    else:
        training_epochs, _ = initialize_model(model, ckpoint_path=args.resume)

    model.eval()

    save_dir = os.path.join(args.results_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    starting_time = time.time()  # starting time for recording total testing time
    for i in range(len(testing_data)):
        temp_starting_time = time.time()  # starting time for recording testing time per image
        image_tiles, seg, name = testing_data[i]

        message_tmp = "[{}] Segmenting {}".format(i, name)
        print(message_tmp)

        prediction = pred_iter(model, image_tiles, sub_size=test_batch_size)
        prediction = torch.max(prediction, 1)[1]
        pred_image = partition.assemble(prediction, is_vote=if_vote)

        # save prediction as image
        if args.results_dir is not "" and i <= 0:
            print("Saving images")
            seg_image_file = name + "_prediction_{}.nii.gz".format(test_config)
            sitk.WriteImage(pred_image, os.path.join(save_dir, seg_image_file))

    print("Total time: {}".format(time.time() - starting_time))


if __name__ == '__main__':
    main()
