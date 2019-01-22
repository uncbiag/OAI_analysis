#!/usr/bin/env python3


import os
import sys
sys.path.append(os.path.realpath(".."))
# import datasets as data3d

import utils.transforms as bio_transform
import utils.datasets as data3d
from model import *
from utils.tools import *

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--results-dir', default='results/MICCAI', type=str, metavar='PATH',
                        help='path to tensorboard log file (default: none)')
    parser.add_argument('--random-seed', default=1234, type=int,
                        help='seed for deterministic random generater')
    parser.add_argument('--GPU', default=1, type=int,
                        help='index of used GPU')
    args = parser.parse_args()

    n_classes = 3
    n_channels = 1

    experiment_name = args.resume.split('/')[1]
    params = experiment_name.split('_')

    # overlap_size = (0, 0, 0)
    overlap_size = (16, 16, 8)
    test_batch_size = int(params[params.index('batch') + 1]) * 4 - 1
    testing_list_file = os.path.realpath("../data/validation.txt")
    # testing_list_file = os.path.realpath("../data/test1.txt")

    # data_dir = os.path.realpath("../data/Nifti_corrected_rescaled")
    # data_dir = os.path.realpath("../data/Nifti_corrected_cropped_rescaled")
    # data_dir = os.path.realpath("../data/Nifti_cropped_rescaled")
    data_dir = os.path.realpath("../data/Nifti_rescaled")
    # data_dir = os.path.realpath("../data/Nifti_rescaled")

    # experiment_name = 'patch_{}_{}_{}_bspline_300'.format(patch_size[0], patch_size[1], patch_size[2])

    patch_size = (int(params[params.index('patch')+1]), int(params[params.index('patch')+2]), int(params[params.index('patch')+3])) # size of 3d training patches cropped from the original image
    orginal_size = (250, 165, 148)

    # experiment_name = "unet_test_200"
    padding_mode = 'reflect'
    if_vote = False
    if_bspline = False
    if_gaussian = False
    if_bilateral = False
    if_log = True
    if_save = False
    if_toRight = 'toRight' in params
    test_config = "{}_{}{}{}{}{}".format(overlap_size[0], padding_mode, '_vote' if if_vote else '',
                                       '_bspline' if if_bspline and 'bspline' in params else '',
                                         '_gaussian' if if_gaussian and 'GaussianBlur' in params else '',
                                       '_bilateral' if if_bilateral and 'Bilateral' in params else '')
    print("Test config: " + test_config)
    partition = bio_transform.Partition(patch_size, overlap_size, padding_mode=padding_mode, mode='pred')
    test_transforms = [partition]

    if 'BN' in params:
        if_BN = True

    if 'bias' in params:
        if_bias = True

    if 'bspline' in params and if_bspline:
        bspline_params = params[params.index('bspline')+1].split('-')
        bspline_order = int(bspline_params[bspline_params.index('order')+1])
        if bspline_params.index('scale') - bspline_params.index('mesh') == 2:
            mesh_size = (int(bspline_params[bspline_params.index('mesh') + 1]),)*3
        if bspline_params.index('scale') - bspline_params.index('mesh') == 4:
            mesh_size = (int(bspline_params[bspline_params.index('mesh')+1]),
                         int(bspline_params[bspline_params.index('mesh')+2]),
                         int(bspline_params[bspline_params.index('mesh')+3]))
        if 'local' in bspline_params :
            mesh_size = tuple(mesh_size[i]*orginal_size[i]//patch_size[i] for i in range(3))
        interpolator = bspline_params[bspline_params.index('interp')+1]
        deform_scale = float(bspline_params[bspline_params.index('scale')+1])
        bspline = bio_transform.RandomBSplineTransform(
            bspline_order=bspline_order, mesh_size=mesh_size, deform_scale=deform_scale, ratio=1.0,
            interpolator=sitk.sitkLinear if interpolator == 'Linear' else sitk.sitkBSpline)
        test_transforms.insert(0, bspline)

    if 'GaussianBlur' in params and if_gaussian:
        gaussian_params = params[params.index('GaussianBlur')+1].split('-')
        blur_ratio = float(gaussian_params[gaussian_params.index('ratio')+1])
        gaussian_var = float(gaussian_params[gaussian_params.index('var')+1])
        gaussian_width = int(gaussian_params[gaussian_params.index('width')+1])
        gaussian_blur = bio_transform.GaussianBlur(
            variance=gaussian_var, maximumKernelWidth=gaussian_width, maximumError=0.9, ratio=blur_ratio)
        test_transforms.insert(0, gaussian_blur)

    if 'Bilateral' in params and if_bilateral:
        bilateral_params = params[params.index('Bilateral')+1].split('-')
        bilateral_ratio = float(bilateral_params[bilateral_params.index('ratio')+1])
        domain_sigma = float(bilateral_params[bilateral_params.index('sigmaD')+1])
        range_sigma = float(bilateral_params[bilateral_params.index('sigmaR')+1])
        number_of_range_gaussian_samples = int(bilateral_params[bilateral_params.index('rangeSamples')+1])
        bilateral_filter = bio_transform.BilateralFilter(ratio=bilateral_ratio, domainSigma=domain_sigma,
                                                         rangeSigma=range_sigma,
                                                         numberOfRangeGaussianSamples=number_of_range_gaussian_samples)
        test_transforms.insert(0, bilateral_filter)

    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = False
    # set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    # set random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np_rand_state = np.random.RandomState(seed=args.random_seed)

    # load training data and validation data
    print("load data from {}".format(data_dir))

    test_transform = transforms.Compose(test_transforms)

    testing_data = data3d.NiftiDataset(testing_list_file, data_dir, "training", transform=test_transform,
                                       to_right=if_toRight)
    # testing_data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=4)

    # build unet
    if 'light1' in params:
        model = UNet_light1(in_channel=n_channels, n_classes=n_classes, BN='BN' in params,
                            bias='bias' in params)
    elif 'light2' in params:
        model = UNet_light2(in_channel=n_channels, n_classes=n_classes, BN='BN' in params,
                            bias='bias' in params)
    elif 'light3' in params:
        model = UNet_light3(in_channel=n_channels, n_classes=n_classes, BN='BN' in params,
                            bias='bias' in params)
    elif 'light4' in params:
        model = UNet_light4(in_channel=n_channels, n_classes=n_classes, BN='BN' in params,
                            bias='bias' in params)
    else:
        model = UNet(in_channel=n_channels, n_classes=n_classes, BN='BN' in params,
                     bias='bias' in params)
    model.cuda()

    # resume checkpoint or initialize
    training_epochs = 0
    if args.resume is None:
        raise ValueError("The path of trained model is needed.")
    else:
        training_epochs, _ = initialize_model(model, ckpoint_path=args.resume)

    # for i, (image, seg, name) in enumerate(testing_data_loader):
    #     print(i, image.shape, seg.shape, name)

    # testing
    model.eval()
    iou_list = []
    dice_FC_list = []
    dice_TC_list = []
    precision_list = []
    recall_list = []

    save_dir = os.path.join(args.results_dir, experiment_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if if_log:
        test_log = open(os.path.join(save_dir, 'test_log.txt'), 'a')
        test_log.write('\n'+"="*50+'\n')
        test_log.write('Testing Model: ' + args.resume + '({} epochs)'.format(training_epochs) + '\n')
        test_log.write('Test data: ' + data_dir + '\n')
        test_log.write('with test config: ' + test_config + '\n')
        test_log.write('\n'+"-"*50+'\n')
        test_log.write('mIOU,' + ' '*3 + 'Dice_FC,' +' '*1 + 'Dice_TC')
        test_log.write('\n'+"-"*50+'\n')

    Left_Dice_FC = 0
    Left_Dice_TC = 0
    Right_Dice_FC = 0
    Right_Dice_TC = 0

    # for i in range(1):
    starting_time = time.time()  # starting time for recording total testing time
    for i in range(len(testing_data)):
        temp_starting_time = time.time()  # starting time for recording testing time per image
        image_tiles, seg, name = testing_data[i]

        message_tmp = "[{}] Testing {}".format(i, name)
        print(message_tmp)

        if if_log:
            test_log.write(message_tmp + '\n')


        # print(get_gpu_memory_map())
        prediction = pred_iter(model, image_tiles, sub_size=test_batch_size)
        # print(get_gpu_memory_map())


        prediction = torch.max(prediction, 1)[1]
        pred_image = partition.assemble(prediction, is_vote=if_vote)
        pred_np = sitk.GetArrayViewFromImage(pred_image)

        performance = metrics.get_multi_metric(np.expand_dims(pred_np, axis=0), np.expand_dims(seg.numpy(), axis=0))
        iou_list.append(performance['label_avg_res']['iou'][0, 0])
        dice_FC_list.append(performance['multi_metric_res']['dice'][0, 1])
        dice_TC_list.append(performance['multi_metric_res']['dice'][0, 2])

        # iou_list.append(metrics.metricEval('iou', pred_np, seg.numpy(), num_labels=3))
        # dice_FC_list.append(metrics.metricEval('dice', pred_np == 1, seg.numpy()==1, num_labels=2))
        # dice_TC_list.append(metrics.metricEval('dice', pred_np == 2, seg.numpy()==2, num_labels=2))
        # # precision_list.append(metrics.metricEval('precision', pred_np, seg.numpy(), num_labels=3))
        # # recall_list.append(metrics.metricEval('recall', pred_np, seg.numpy(), num_labels=3))

        print("mIOU: {:.4f}, FC Dice: {:.4f}, TC Dice: {:.4f}, time:{:.2f}"
              .format(iou_list[i], dice_FC_list[i], dice_TC_list[i], time.time()-temp_starting_time))
        if if_log:
            test_log.write('{:.4f}, {:.4f}, {:.4f}'.format(iou_list[i], dice_FC_list[i], dice_TC_list[i]) + '\n')

        # save prediction as image
        if if_save:
            print("Saving images")
            seg_image_file = name + "_prediction_{}.nii.gz".format(test_config)
            sitk.WriteImage(pred_image, os.path.join(save_dir, seg_image_file))

        if 'LEFT' in name:
            Left_Dice_FC += dice_FC_list[i]
            Left_Dice_TC += dice_TC_list[i]
        if 'RIGHT' in name:
            Right_Dice_FC += dice_FC_list[i]
            Right_Dice_TC += dice_TC_list[i]


    avg_iou = np.mean(np.asarray(iou_list))
    std_iou = np.std(np.asarray(iou_list))
    avg_dice_FC = np.mean(np.asarray(dice_FC_list))
    std_dice_FC = np.std(np.asarray(dice_FC_list))
    avg_dice_TC = np.mean(np.asarray(dice_TC_list))
    std_dice_TC = np.std(np.asarray(dice_TC_list))

    message_tmp = "Average: {:.4f}, {:.4f}, {:.4f} (avg Dice {:.4f})\\n" \
                  "STD: {:.4f}, {:.4f}, {:.4f}\n" \
                  "Time: {:.2f}" \
        .format(avg_iou, avg_dice_FC, avg_dice_TC, (avg_dice_FC + avg_dice_TC) / 2, std_iou, std_dice_FC, std_dice_TC,
                time.time() - starting_time)
    if if_log:
        test_log.write(message_tmp + '\n')
        test_log.close()
    print(message_tmp)

    print("Left_FC:{} Left_TC:{} Right_FC:{} Right_TC:{}".
          format(Left_Dice_FC / len(testing_data) * 2, Left_Dice_TC / len(testing_data) * 2,
                 Right_Dice_FC / len(testing_data) * 2,
                 Right_Dice_TC / len(testing_data) * 2))

if __name__ == '__main__':
    main()
