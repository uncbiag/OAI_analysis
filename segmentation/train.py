"""
Train 3d segmentation model
"""
import os
import sys
import time
import datetime
import argparse

import numpy as np
import SimpleITK as sitk
import torch
# import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

# import torch.nn.functional as F

sys.path.append(os.path.realpath(".."))
sys.path.append(os.path.realpath("."))
import misc.image_transforms as bio_transform
from misc.module_parameters import save_dict_to_json
import segmentation.datasets as data3d
from segmentation.networks import get_network, get_available_networks
from segmentation.loss import get_loss_function
from segmentation.utils import initialize_model, weight_from_truth, save_checkpoint
from segmentation.visualize import make_segmentation_image_summary
import segmentation.metrics as metrics


def train_segmentation(model, training_data_loader, validation_data, optimizer, criterion, device, config,
                       scheduler=None):
    # log writer
    now = datetime.datetime.now()
    now_date = "{:02d}{:02d}{:02d}".format(now.month, now.day, now.year)
    now_time = "{:02d}{:02d}{:02d}".format(now.hour, now.minute, now.second)

    total_validation_time = 0
    if not config['debug_mode']:
        writer = SummaryWriter(os.path.join(args.logdir, now_date, config['experiment_name'] + '_' + now_time))

    ckpoint_name = os.path.join(config['experiment_name'], now_date + '_' + now_time) if not config[
        'debug_mode'] else "debug"

    if not os.path.isdir(os.path.join(config['ckpoint_dir'], ckpoint_name)):
        os.makedirs(os.path.join(config['ckpoint_dir'], ckpoint_name))
    save_dict_to_json(config, os.path.join(config['ckpoint_dir'], ckpoint_name, "train_config.json"))

    # resume checkpoint or initialize
    finished_epochs, best_score = initialize_model(model, optimizer, args.resume)
    current_epoch = finished_epochs + 1

    iters_per_epoch = len(training_data_loader.dataset)
    print("Start Training:")
    while current_epoch <= config['n_epochs']:
        running_loss = 0.0
        is_best = False
        start_time = time.time()  # log running time

        if not config['lr_mode'] == 'const' and not config['lr_mode'] == 'plateau':
            scheduler.step(epoch=current_epoch)

        for i, (images, truths, name) in enumerate(training_data_loader):
            global_step = (current_epoch - 1) * iters_per_epoch + (i + 1) * config[
                'batch_size']  # current globel step

            model.train()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(images.to(device))

            # start_time = time.time()
            # print(time.time() - start_time)

            loss = criterion(output, truths.to(device))
            # del output_flat, truths_flat
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()  # average loss over 10 batches
            if i % config['print_batch_period'] == config['print_batch_period'] - 1:  # print every 10 mini-batches
                duration = time.time() - start_time
                print('Epoch: {:0d} [{}/{} ({:.0f}%)] loss: {:.3f} lr:{} ({:.3f} sec/batch) {}'.format
                      (current_epoch, (i + 1) * config['batch_size'], iters_per_epoch,
                       (i + 1) * config['batch_size'] / iters_per_epoch * 100,
                       running_loss / config['print_batch_period'] if i > 0 else running_loss,
                       optimizer.param_groups[0]['lr'],
                       duration / config['print_batch_period'],
                       datetime.datetime.now().strftime("%D %H:%M:%S")
                       ))
                if not config['debug_mode']:
                    writer.add_scalar('loss/training', running_loss / config['print_batch_period'],
                                      global_step=global_step)  # data grouping by `slash`
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'],
                                      global_step=global_step)  # data grouping by `slash`
                running_loss = 0.0
                start_time = time.time()  # log running time

            if config['debug_mode']:
                image_summary = make_segmentation_image_summary(images, truths, output.cpu())

        if not config['debug_mode'] and current_epoch % config['save_ckpts_epoch_period'] == 0:
            image_summary = make_segmentation_image_summary(images, truths, output.cpu())
            writer.add_image("training", image_summary, global_step=global_step)

        # validation
        with torch.no_grad():
            if current_epoch % config['valid_epoch_period'] == 0:
                model.eval()
                dice_FC = 0
                dice_TC = 0
                start_time = time.time()  # log running time
                for j in range(len(validation_data) if best_score > 0.7 else 2):
                    image_tiles, truths, name = validation_data[j]

                    valid_batch_size = config['valid_batch_size']
                    outputs = []  # a list of lists that store output at each step
                    # for i in range(1):
                    for i in range(0, np.ceil(image_tiles.size()[0] / valid_batch_size).astype(int)):
                        temp_input = image_tiles.narrow(0, valid_batch_size * i,
                                                        valid_batch_size if valid_batch_size * (i + 1) <=
                                                                            image_tiles.size()[
                                                                                0]
                                                        else image_tiles.size()[0] - valid_batch_size * i)

                        # predict through model 1
                        temp_output = model(temp_input.to(device)).cpu().data

                        outputs.append(temp_output)
                        del temp_input

                    predictions = torch.cat(outputs, dim=0)

                    pred_assemble = validation_data.transform.transforms[0].assemble(torch.max(predictions, 1)[1],
                                                                                     if_itk=False)

                    dice_FC += metrics.metricEval('dice', pred_assemble == 1, truths.numpy() == 1,
                                                  num_labels=2)
                    dice_TC += metrics.metricEval('dice', pred_assemble == 2, truths.numpy() == 2,
                                                  num_labels=2)
                dice_FC = dice_FC / (j + 1)
                dice_TC = dice_TC / (j + 1)
                dice_avg = (dice_FC + dice_TC) / 2

            if config['lr_mode'] == 'plateau':
                scheduler.step(dice_avg, epoch=current_epoch)

            is_best = False
            if dice_avg > best_score:
                is_best = True
                best_score = dice_avg

            if not config['debug_mode']:
                # writer.add_scalar('loss/validation', valid_loss_avg,
                #                   global_step=global_step)  # data grouping by `slash`
                writer.add_scalar('validation/dice_avg', dice_avg, global_step=global_step)
                writer.add_scalar('validation/dice_FC', dice_FC, global_step=global_step)
                writer.add_scalar('validation/dice_TC', dice_TC, global_step=global_step)

            # image_summary = make_segmentation_image_summary(images, truths, output)
            # writer.add_image("validation", image_summary, global_step=global_step)

            print('Epoch: {:0d} Validation: Dice Avg: {:.3f} Dice_FC: {:.3f} Dice_TC:{:.3f} ({:.3f} sec) {}'.format
                  (current_epoch, dice_avg, dice_FC,
                   dice_TC, time.time() - start_time,
                   datetime.datetime.now().strftime("%D %H:%M:%S")))
            total_validation_time += time.time() - start_time

        if not config['debug_mode'] and current_epoch % config['save_ckpts_epoch_period'] == 0:
            save_checkpoint({'epoch': current_epoch,
                             'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'best_score': best_score},
                            is_best, args.save, config['experiment_name'] + '_' + now_date + '_' + now_time)

        current_epoch += 1

    writer.close()
    print('Finished Training: {}_{}_{}'.format(config['experiment_name'], now_date, now_time))
    print('Total validation time: {}'.str(datetime.timedelta(seconds=total_validation_time)))


def main(args):
    class_ratio = np.array([4, 4, 1])  # ratio of number of voxels with differenct class label
    data_root = "/playpen/zhenlinx/Data/OAI_segmentation"
    config = dict(
        debug_mode=args.debug,
        device='cpu' if args.GPU == -1 else 'cuda',

        ckpoint_dir='./ckpoints',
        log_dir='../log',
        resume_dir=None,

        # data config
        training_list_file=os.path.join(data_root, "train1.txt"),
        validation_list_file=os.path.join(data_root, "validation1.txt"),
        data_dir=os.path.join(data_root, "Nifti_rescaled"),
        valid_data_dir=os.path.join(data_root, "Nifti_rescaled"),
        patch_size=(128, 128, 32),
        # size of 3d patch cropped from the original image (250, 165, 148)X -> (250, 183, 152)
        # patch_size=(128, 128, 96),
        # patch_size=(200, 200, 160),
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

        model='UNet',  # UNet/UNet_light/UNet_light2
        model_setting=dict(n_classes=3,
                           in_channels=1,
                           bias=True,
                           BN=False,
                           ),

        # set random seed
        # torch.manual_seed(args.random_seed)
        # torch.cuda.manual_seed(args.random_seed)
        rand_seed=args.random_seed,

        # weighted loss function
        # if_weight_loss=False,
        # weight_mode='const',
        # weight_const=torch.FloatTensor(1 / class_ratio / np.sum(1 / class_ratio)),

        # config loss and optimizer
        # if_dice_loss= False,
        loss='Dice',
        loss_setting=dict(),

        learning_rate=1e-3,
        lr_mode='multiStep',  # const/plateau/...

    )
    # optimizer scheduler setting
    if config['lr_mode'] == 'plateau':
        plateau_threshold = 0.003
        plateau_patience = 100
        plateau_factor = 0.2

    config['print_batch_period'] = max(120 // config['batch_size'] // 2, 1)

    if_to_left = True

    # essential transforms
    sample_threshold = (0.01, 0.02)
    if config['patch_size'][2] > 32:
        sample_threshold = (0.005, 0.01)
    if config['patch_size'][0] > 200:
        sample_threshold = (0.0, 0.0)

    # sample_threshold = (0.0, 0.0)
    balanced_random_crop = bio_transform.BalancedRandomCrop(config['patch_size'], threshold=sample_threshold)
    sitk_to_tensor = bio_transform.SitkToTensor()
    partition = bio_transform.Partition(config['patch_size'], overlap_size=(16, 16, 8))

    # set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    # load training data and validation data
    print("Initializing dataloader")

    # set up data loader

    train_transforms = [balanced_random_crop, sitk_to_tensor]
    valid_transforms = [partition]

    if config['debug_mode']:
        print("Debug mode")
        config['valid_epoch_period'] = 1

    config['experiment_name'] = '{}{}{}{}{}{}{}{}{}{}'.format(
        '{}{}{}_'.format(config['model'], '_bias' if config['model_setting']['bias'] else '',
                         '_BN' if config['model_setting']['BN'] else ''),
        os.path.basename(config['data_dir']),
        '_toLeft' if if_to_left else '',
        '_{}'.format(os.path.basename(config['training_list_file']).split('.')[0]),
        '_patch_{}_{}_{}'.format(config['patch_size'][0], config['patch_size'][1], config['patch_size'][2]),
        '_batch_{}'.format(config['batch_size']),
        '_sample_{}-{}'.format(sample_threshold[0], sample_threshold[1]),
        '_{}'.format(config['loss']),
        '_' + ('_').join(["{}-{}".format(key, value) for key, value in config["loss_setting"].items()]) if config[
            "loss_setting"] else '',
        '_lr_{}'.format(config['learning_rate']),
        '_scheduler_{}'.format(config['lr_mode']) if not config['lr_mode'] == 'const' else ''
    )

    print("Training {}".format(config['experiment_name']))

    train_transform = transforms.Compose(train_transforms)

    # valid_transform = transforms.Compose([bio_transform.RandomCrop(config['patch_size'], threshold=0,
    #                                                                random_state=config['np_rand_state']), sitk_to_tensor])

    valid_transform = transforms.Compose(valid_transforms)

    training_data = data3d.NiftiDataset(config['training_list_file'], config['data_dir'], mode="training",
                                        preload=False if config['debug_mode'] else True, transform=train_transform)
    training_data_loader = DataLoader(training_data, batch_size=config['batch_size'],
                                      shuffle=True, num_workers=4, pin_memory=False)

    validation_data = data3d.NiftiDataset(config['validation_list_file'], config['valid_data_dir'], mode="training",
                                          preload=False if config['debug_mode'] else True, transform=valid_transform,
                                          )
    # validation_data_loader = DataLoader(validation_data, batch_size=4,
    #                                     shuffle=True, num_workers=2, pin_memory=True)

    # set device
    device = torch.device(config['device'])

    # build unet
    model = get_network(config['model'])(**config['model_setting'])
    model.to(device)

    # loss function
    criterion = get_loss_function(loss_name=config['loss'])(**config['loss_setting'])

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
    train_segmentation(model, training_data_loader, validation_data, optimizer, criterion, device, config,
                       scheduler=scheduler if not config['lr_mode'] == 'const' else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',
                        default=
                        # "ckpoints/UNet_light4_Nifti_rescaled_patch_128_128_32_batch_12_sample_0.01-0.02_cross_entropy_lr_0.0001_01302018_105659/checkpoint.pth.tar", type=str, metavar='PATH',
                        "",
                        help='path to latest checkpoint (default: none)')  # logs/unet_test_checkpoint.pth.tar
    parser.add_argument('--ckpointDir', default='ckpoints', type=str, metavar='PATH',
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
    main(args)
