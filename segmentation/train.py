import os
import sys

sys.path.append(os.path.realpath(".."))

"""
Train a 3d unet
"""
# import datasets as data3d
import utils.transforms as bio_transform
import utils.datasets as data3d
from model import *
from utils.tools import *


def test_func():
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',
                        default=
                        # "ckpoints/UNet_light4_Nifti_rescaled_patch_128_128_32_batch_12_sample_0.01-0.02_cross_entropy_lr_0.0001_01302018_105659/checkpoint.pth.tar", type=str, metavar='PATH',
                        "",
                        help='path to latest checkpoint (default: none)')  # logs/unet_test_checkpoint.pth.tar
    parser.add_argument('--save', default='ckpoints', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--logdir', default='logs', type=str, metavar='PATH',
                        help='path to tensorboard log file (default: none)')
    parser.add_argument('--GPU', default=1, type=int,
                        help='index of used GPU')
    parser.add_argument('--random-seed', default=1234, type=int,
                        help='seed for deterministic random generater')
    parser.add_argument('--debug', default=1, type=int,
                        help='if debug mode')
    args = parser.parse_args()

    class_ratio = np.array([4, 4, 1])  # ratio of number of voxels with differenct class label

    config = dict(
        debug_mode=args.debug,

        training_list_file=os.path.realpath("../data/train1.txt"),
        validation_list_file=os.path.realpath("../data/validation1.txt"),
        data_dir=os.path.realpath("../data/Nifti_rescaled"),
        # data_dir = os.path.realpath("../data/Nifti_rescaled_cropped"),
        # data_dir=os.path.realpath("../data/Nifti_corrected_cropped_rescaled"),
        valid_data_dir=os.path.realpath("../data/Nifti_rescaled"),
        # patch_size=(128, 128, 32),  # size of 3d patch cropped from the original image (250, 165, 148)X -> (250, 183, 152)
        # patch_size=(128, 128, 96),
        patch_size=(200, 200, 160),
        # patch_size=(250, 165, 148),
        # patch_size=(248, 164, 148),
        # patch_size=(128, 128, 64),
        # patch_size = (192, 192, 96),
        n_epochs=600,
        batch_size=1,
        valid_batch_size=1,
        print_batch_period=10,
        valid_epoch_period=10,
        save_ckpts_epoch_period=10,

        model='UNet_light2',  # UNet/UNet_light/UNet_light2

        n_classes=3,
        n_channels=1,
        if_bias=1,
        if_batchnorm=0,

        # set random seed
        # torch.manual_seed(args.random_seed)
        # torch.cuda.manual_seed(args.random_seed)
        np_rand_state= np.random.RandomState(seed=args.random_seed),

        # weighted loss function
        if_weight_loss=False,
        weight_mode='const',
        weight_const=torch.FloatTensor(1/class_ratio/np.sum(1/class_ratio)),

        # config loss and optimizer
        # if_dice_loss= False,
        loss='cross_entropy',  # cross_entropy/dice/focal_loss
        learning_rate=1e-3,
        lr_mode='multiStep',  # const/plateau/...
    )

    # optimizer scheduler setting
    if config['lr_mode'] == 'plateau':
        plateau_threshold = 0.003
        plateau_patience = 100
        plateau_factor = 0.2

    config['print_batch_period'] = max(120 // config['batch_size'] // 2, 1)

    if_to_right = False
    if_rigid_transform = False
    if_bspline_deform = False  # if use random bspline transformation for data argumentation
    if_gaussian_blur = False
    if_bilateral = False

    # essential transforms
    sample_threshold = (0.01, 0.02)
    if config['patch_size'][2] > 32:
        sample_threshold = (0.005, 0.01)
    if config['patch_size'][0] > 200:
        sample_threshold = (0.0, 0.0)

    sample_threshold = (0.0, 0.0)
    balanced_random_crop = bio_transform.BalancedRandomCrop(config['patch_size'], threshold=sample_threshold)
    sitk_to_tensor = bio_transform.SitkToTensor()
    partition = bio_transform.Partition(config['patch_size'], overlap_size=(16, 16, 8))

    # rigid transform
    transition = (0.5, 0.5, 0.5)
    rotation = (0, 0, 0)
    rigid_ratio = 0.5
    rigid_mode = 'both'
    rigid_transform = bio_transform.RandomRigidTransform(ratio=rigid_ratio, translation=transition,
                                                         rotation_angles=rotation, mode=rigid_mode)

    # bspline setting
    bspline_order = 3
    mesh_size = (2, 2, 2)
    deform_scale = 15.0
    deform_ratio = 0.5
    interpolator = "BSpline"  # interpolator used for resampling
    deform_target = "padded"
    # 'global': bspline deformation is on un-cropped data
    # 'padded': on a pre-padded patch
    # 'local' dirrectly on cropped patch
    bspline_transform = bio_transform.RandomBSplineTransform(
        mesh_size=mesh_size, bspline_order=bspline_order, deform_scale=deform_scale,
        ratio=deform_ratio, interpolator=sitk.sitkBSpline if interpolator == "BSpline" else sitk.sitkLinear)

    # Gaussian Blur
    blur_ratio = 1.0
    gaussian_var = 0.5

    gaussian_width = 1
    gaussian_blur = bio_transform.GaussianBlur(
        variance=gaussian_var, maximumKernelWidth=gaussian_width, maximumError=0.9, ratio=blur_ratio)

    # Bilateral Filtering
    bilateral_ratio = 1.0
    domain_sigma = 0.2
    range_sigma = 0.06
    number_of_range_gaussian_samples = 50
    bilateral_filter = bio_transform.BilateralFilter(ratio=bilateral_ratio, domainSigma=domain_sigma,
                                                     rangeSigma=range_sigma,
                                                     numberOfRangeGaussianSamples=number_of_range_gaussian_samples)

    # set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    # load training data and validation data
    print("Initializing dataloader")

    # set up data loader

    train_transforms = [balanced_random_crop, sitk_to_tensor]
    valid_transforms = [partition]

    if if_bspline_deform:
        if deform_target == 'global':
            train_transforms.insert(0, bspline_transform)
        elif deform_target == 'local':
            train_transforms.insert(1, bspline_transform)
        elif deform_target == 'padded':
            train_transforms.insert(0, bspline_transform)
            pre_padded_size = tuple(config['patch_size'][i] + int(deform_scale) * 2 for i in range(3))
            pre_random_crop = bio_transform.BalancedRandomCrop(pre_padded_size)
            post_crop = bio_transform.RandomCrop(config['patch_size'], threshold=-1)
            train_transforms = [pre_random_crop, bspline_transform, post_crop, sitk_to_tensor]

    if if_rigid_transform:
        train_transforms.insert(0, rigid_transform)

    if if_gaussian_blur:
        train_transforms.insert(-1, gaussian_blur)

    if if_bilateral:
        train_transforms.insert(1, bilateral_filter)

    if config['debug_mode']:
        print("Debug mode")
        config['valid_epoch_period'] = 1

    config['experiment_name'] = '{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}'.format(
        '{}{}{}_'.format(config['model'], '_bias' if config['if_bias'] else '',
                         '_BN' if config['if_batchnorm'] else ''),
        os.path.basename(config['data_dir']),
        '_toRight' if if_to_right else '',
        '_{}'.format(os.path.basename(config['training_list_file']).split('.')[0]),
        '_patch_{}_{}_{}'.format(config['patch_size'][0], config['patch_size'][1], config['patch_size'][2]),
        '_batch_{}'.format(config['batch_size']),
        '_sample_{}-{}'.format(sample_threshold[0], sample_threshold[1]),
        '_rigid_rotation-{}-{}-{}-trans-{}-{}-{}-ratio-{}-mode-{}'.format(
            rotation[0], rotation[1], rotation[2], transition[0], transition[1], transition[2], rigid_ratio,
            rigid_mode) if if_rigid_transform else '',
        '_bspline_{}-order-{}-mesh-{}-{}-{}-scale-{}-ratio-{}-interp-{}-ignoreZ'.format(
            deform_target, bspline_order, mesh_size[0], mesh_size[1], mesh_size[2],
            deform_scale, deform_ratio, interpolator) if if_bspline_deform else '',
        '_GaussianBlur_var-{}-width-{}-ratio-{}'.format(gaussian_var, gaussian_width,
                                                        blur_ratio) if if_gaussian_blur else '',
        '_Bilateral_sigmaD-{}-sigmaR-{}-rangeSamples-{}-ratio-{}'.format(
            domain_sigma, range_sigma, number_of_range_gaussian_samples, bilateral_ratio) if if_bilateral else '',
        '_{}'.format(config['loss']),
        '_weighted-loss_{}{}'.format(config['weight_mode'],
                                     '-{}-{}-{}'.format(class_ratio[0], class_ratio[1], class_ratio[2])
                                     if config['weight_mode'] == 'const' else '')
        if config['if_weight_loss'] else '',
        '_lr_{}'.format(config['learning_rate']),
        '_scheduler_{}'.format(config['lr_mode']) if not config['lr_mode'] == 'const' else ''
    )

    print("Training {}".format(config['experiment_name']))

    train_transform = transforms.Compose(train_transforms)

    # valid_transform = transforms.Compose([bio_transform.RandomCrop(config['patch_size'], threshold=0,
    #                                                                random_state=config['np_rand_state']), sitk_to_tensor])

    valid_transform = transforms.Compose(valid_transforms)


    training_data = data3d.NiftiDataset(config['training_list_file'], config['data_dir'], mode="training",
                                        preload=False if config['debug_mode'] else True, transform=train_transform,
                                        to_right=if_to_right)
    training_data_loader = DataLoader(training_data, batch_size=config['batch_size'],
                                      shuffle=True, num_workers=4 if if_bspline_deform else 4, pin_memory=False, )

    validation_data = data3d.NiftiDataset(config['validation_list_file'], config['valid_data_dir'], mode="training",
                                          preload=False if config['debug_mode'] else True, transform=valid_transform,
                                          to_right=if_to_right)
    # validation_data_loader = DataLoader(validation_data, batch_size=4,
    #                                     shuffle=True, num_workers=2, pin_memory=True)

    # build unet
    model_type = globals()[config['model']]
    model = model_type(in_channel=config['n_channels'], n_classes=config['n_classes'], bias=config['if_bias'],
                       BN=config['if_batchnorm'])
    model.cuda()

    # criterion = nn.CrossEntropyLoss().cuda()  # loss function
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    if config['lr_mode'] == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                   patience=plateau_patience // config['valid_epoch_period'],
                                                   factor=plateau_factor, verbose=True, threshold_mode='abs',
                                                   threshold=plateau_threshold,
                                                   min_lr=1e-5)

    if config['lr_mode'] == 'multiStep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, config['n_epochs'] - 100], gamma=0.1)

    # training
    train_model(model, training_data_loader, validation_data, optimizer, args, config,
                scheduler=scheduler if not config['lr_mode'] == 'const' else None)




if __name__ == '__main__':
    main()
