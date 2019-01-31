#!/usr/bin/env python
"""
Functions of PyTorch tensor operations
Created by zhenlinx on 1/20/19
"""


class SegMaskToOneHot:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, sample):
        seg = sample['segmentation']
        seg_one_hot = self.one_mask_to_one_hot(seg)
        sample['segmentation_onehot'] = seg_one_hot
        return sample

    def one_mask_to_one_hot(self, mask):
        """

        :param mask:1xDxMxN
        :return: one-hot mask CxDxMxN
        """
        one_hot_shape = list(mask.shape)
        one_hot_shape[0] = self.n_classes
        mask_one_hot = torch.zeros(one_hot_shape).to(mask.device)
        mask_one_hot.scatter_(0, mask.long(), 1)
        return mask_one_hot


def mask_to_one_hot(mask, n_classes):
    """
    Convert a segmentation mask to one-hot coded tensor
    :param mask: mask tensor of size Bx1xDxMxN
    :param n_classes: number of classes
    :return: one_hot: BxCxDxMxN
    """
    one_hot_shape = list(mask.shape)
    one_hot_shape[1] = n_classes
    mask_one_hot = torch.zeros(one_hot_shape).to(mask.device)

    mask_one_hot.scatter_(1, mask.long(), 1)

    return mask_one_hot