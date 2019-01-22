import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
from itertools import repeat
import numpy as np
from utils.transforms import mask_to_one_hot

# Intersection = dot(A, B)
# Union = dot(A, A) + dot(B, B)
# The Dice loss function is defined as
# 1/2 * intersection / union
#
# The derivative is 2[(union * target - 2 * intersect * input) / union^2]

# TODO: dice loss for multi-class


class DiceLoss(Function):
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, input, target, save=True):
        if save:
            self.save_for_backward(input, target)
        eps = 0.000001
        _, result_ = input.max(1)
        result_ = torch.squeeze(result_)
        if input.is_cuda:
            result = torch.cuda.FloatTensor(result_.size())
            self.target_ = torch.cuda.FloatTensor(target.size())
        else:
            result = torch.FloatTensor(result_.size())
            self.target_ = torch.FloatTensor(target.size())
        result.copy_(result_)
        self.target_.copy_(target)
        target = self.target_
        #       print(input)
        intersect = torch.dot(result, target)
        # binary values so sum the same as sum of squares
        result_sum = torch.sum(result)
        target_sum = torch.sum(target)
        union = result_sum + target_sum + (2 * eps)

        # the target volume can be empty - so we still want to
        # end up with a score of 1 if the result is 0/0
        IoU = intersect / union
        # print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
        #     union, intersect, target_sum, result_sum, 2*IoU))
        out = torch.FloatTensor(1).fill_(2 * IoU)
        self.intersect, self.union = intersect, union
        return out

    def backward(self, grad_output):
        input, _ = self.saved_tensors
        intersect, union = self.intersect, self.union
        target = self.target_
        gt = torch.div(target, union)
        IoU2 = intersect / (union * union)
        pred = torch.mul(input[:, 1], IoU2)
        dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
        grad_input = torch.cat((torch.mul(dDice, -grad_output[0]),
                                torch.mul(dDice, grad_output[0])), 0)
        return grad_input, None


def dice_loss(input, target):
    return DiceLoss()(input, target)


def dice_error(input, target):
    eps = 0.000001
    _, result_ = input.max(1)
    result_ = torch.squeeze(result_)
    if input.is_cuda:
        result = torch.cuda.FloatTensor(result_.size())
        target_ = torch.cuda.FloatTensor(target.size())
    else:
        result = torch.FloatTensor(result_.size())
        target_ = torch.FloatTensor(target.size())
    result.copy_(result_.data)
    target_.copy_(target.data)
    target = target_
    intersect = torch.dot(result, target)

    result_sum = torch.sum(result)
    target_sum = torch.sum(target)
    union = result_sum + target_sum + 2 * eps
    intersect = np.max([eps, intersect])
    # the target volume can be empty - so we still want to
    # end up with a score of 1 if the result is 0/0
    IoU = intersect / union
    #    print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
    #        union, intersect, target_sum, result_sum, 2*IoU))
    return 2 * IoU


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, soft_max=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.soft_max = soft_max

    def forward(self, inputs, targets):
        """

        :param inputs: Bxn_classxXxYxZ
        :param targets: Bx.....  , range(0,n_class)
        :return:
        """
        # squeeze the targets from Bx1x...to Bx...
        # if len(inputs.shape) == len(targets.shape) and targets.shape[1] == 1:
        #     targets = targets.squeeze(1)


        if len(inputs.shape) > 2 and len(targets.shape) > 1:
            inputs = inputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, inputs.size(1))
            targets = targets.view(-1)

        targets =targets.long()

        # TODO avoid softmax in focal loss
        if self.soft_max:
            P = F.softmax(inputs, dim=1)
        else:
            P = inputs


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()

        alpha = self.alpha[targets.data.view(-1)].view(-1)

        log_p = - F.cross_entropy(inputs, targets, reduce=False)
        probs = F.nll_loss(P, targets, reduce=False)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


class FocalLoss2(nn.Module):
    """
    https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
    """

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = one_hot(target, input.size(-1))
        logit = F.softmax(input)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit)  # cross entropy
        loss = loss * (1 - logit) ** self.gamma  # focal loss

        return loss.sum()


def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)

    return mask.scatter_(1, index, ones)


"""
segmentation loss
"""


# class DiceLoss(nn.Module):
#     def initialize(self, class_num, weight=None):
#         self.class_num = class_num
#         self.class_num = class_num
#         if weight is None:
#             self.weight = torch.ones(class_num, 1) / self.class_num
#         else:
#             self.weight = weight
#         self.weight = torch.squeeze(self.weight)
#
#     def forward(self, input, target, inst_weights=None, train=None):
#         """
#         input is a torch variable of size BatchxnclassesxHxWxD representing log probabilities for each class
#         target is a Bx....   range 0,1....N_label
#         """
#         in_sz = input.size()
#         from functools import reduce
#         extra_dim = reduce(lambda x, y: x * y, in_sz[2:])
#         targ_one_hot = torch.zeros(in_sz[0], in_sz[1], extra_dim).cuda()
#         targ_one_hot.scatter_(1, target.view(in_sz[0], 1, extra_dim), 1.)
#         target = targ_one_hot.view(in_sz).contiguous()
#         probs = F.softmax(input, dim=1)
#         num = probs * target
#         num = num.view(num.shape[0], num.shape[1], -1)
#         num = torch.sum(num, dim=2)
#
#         den1 = probs  # *probs
#         den1 = den1.view(den1.shape[0], den1.shape[1], -1)
#         den1 = torch.sum(den1, dim=2)
#
#         den2 = target  # *target
#         den2 = den1.view(den2.shape[0], den2.shape[1], -1)
#         den2 = torch.sum(den2, dim=2)
#         # print("den1:{}".format(sum(sum(den1))))
#         # print("den2:{}".format(sum(sum(den2/den1))))
#
#         dice = 2 * (num / (den1 + den2))
#         dice = self.weight.expand_as(dice) * dice
#         dice_eso = dice
#         # dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg
#         dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz
#         return dice_total


# class GeneralizedDiceLoss(nn.Module):
#     def initialize(self, class_num, weight=None):
#         self.class_num = class_num
#         if weight is None:
#             self.weight = torch.ones(class_num, 1)
#         else:
#             self.weight = weight
#
#         self.weight = torch.squeeze(self.weight)
#
#     def forward(self, input, target, inst_weights=None, train=None):
#         """
#         input is a torch variable of size BatchxnclassesxHxWxD representing log probabilities for each class
#         target is a Bx....   range 0,1....N_label
#         """
#         in_sz = input.size()
#         from functools import reduce
#         extra_dim = reduce(lambda x, y: x * y, in_sz[2:])
#         targ_one_hot = torch.zeros(in_sz[0], in_sz[1], extra_dim).cuda()
#         targ_one_hot.scatter_(1, target.view(in_sz[0], 1, extra_dim), 1.)
#         target = targ_one_hot.view(in_sz).contiguous()
#         probs = F.softmax(input, dim=1)
#         num = probs * target
#         num = num.view(num.shape[0], num.shape[1], -1)
#         num = torch.sum(num, dim=2)  # batch x ch
#
#         den1 = probs
#         den1 = den1.view(den1.shape[0], den1.shape[1], -1)
#         den1 = torch.sum(den1, dim=2)  # batch x ch
#
#         den2 = target
#         den2 = den1.view(den2.shape[0], den2.shape[1], -1)
#         den2 = torch.sum(den2, dim=2)  # batch x ch
#         # print("den1:{}".format(sum(sum(den1))))
#         # print("den2:{}".format(sum(sum(den2/den1))))
#         weights = self.weight.expand_as(den1)
#
#         dice = 2 * (torch.sum(weights * num, dim=1) / torch.sum(weights * (den1 + den2), dim=1))
#         dice_eso = dice
#         # dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg
#         dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz
#
#         return dice_total


class DiceLossOnLabel(nn.Module):
    """Dice loss from two inputs of segmentation masks(different with between a mask and a probability map)"""

    def __init__(self, n_class=None, eps=10e-6):
        super(DiceLossOnLabel, self).__init__()
        self.n_class = n_class
        self.eps = eps
        # self.source_one_hot = nn.Parameter()
        # self.target_one_hot = nn.Parameter()

    def forward(self, source, target, weight_type='Uniform', average=True):
        """
        :param source: Tensor of size Bx1xDxMxN
        :param target: Tensor of size Bx1xDxMxN
        :return:
        """
        assert source.shape == target.shape

        if self.n_class is None:
            self.n_class = max(torch.unique(target).max(), torch.unique(source).max()).long().item() + 1

        mask_shape = list(target.shape)
        source_one_hot = mask_to_one_hot(source.view(mask_shape[0], mask_shape[1], -1), self.n_class)
        target_one_hot = mask_to_one_hot(target.view(mask_shape[0], mask_shape[1], -1), self.n_class)

        # does not consider background
        source_one_hot = source_one_hot[:, 1:, :]
        target_one_hot = target_one_hot[:, 1:, :]

        #
        source_volume = source_one_hot.sum(2)
        target_volume = target_one_hot.sum(2)

        if weight_type == 'Simple':
            weights = target_volume.float().reciprocal()
            weights = torch.where(torch.isinf(weights), torch.ones_like(weights), weights)
        elif weight_type == 'Uniform':
            weights = torch.ones(mask_shape[0], mask_shape[1])

        intersection = source_one_hot * target_one_hot
        scores = (2. * intersection.sum(2).float() * weights) / (
                weights * (source_volume.float() + target_volume.float()) + self.eps)

        return 1 - scores.mean()


class DiceLossMultiClass(nn.Module):
    """Dice loss from two inputs of segmentation masks(different with between a mask and a probability map)"""

    def __init__(self, n_class=None, weight_type='Simple', no_bg=False, eps=1e-10):
        super(DiceLossMultiClass, self).__init__()
        self.weight_type = weight_type
        self.n_class = n_class
        self.eps = eps
        self.no_bg=no_bg
        # self.source_one_hot = nn.Parameter()
        # self.target_one_hot = nn.Parameter()

    def forward(self, source, target):
        """

        :param source: Tensor of size BxCxDxMxN, One-hot encoding mask /class-wise probability of prediction
        :param target: Tensor, ground truth mask when of size Bx1xDxMxN;
                             One-hot encoding mask /class-wise probability of prediction when of size BxCxDxMxN
        :return:
        """
        assert source.shape[0] == target.shape[0]
        assert source.shape[2:] == target.shape[2:]

        if self.n_class is None:
            self.n_class = max(torch.unique(target).max(), torch.unique(source).max()).long().item() + 1

        shape = list(source.shape)

        # flat the spatial dimensions
        source_flat = source.view(shape[0], shape[1], -1)

        # flat the spatial dimensions and transform it into one-hot coding
        if target.shape[1] == 1:
            target_flat = mask_to_one_hot(target.view(shape[0], 1, -1), self.n_class)
        elif target.shape[1] == shape[1]:
            target_flat = target.view(shape[0], shape[1], -1)
        else:
            ValueError("Incorrect size of target tensor: {}, should be {} or []".format(target.shape, shape,
                                                                                        shape[:1] + [1, ] + shape[2:]))

        # does not consider background
        if self.no_bg:
            source_flat = source_flat[:, 1:, :]
            target_flat = target_flat[:, 1:, :]

        #
        source_volume = source_flat.sum(2)
        target_volume = target_flat.sum(2)

        if self.weight_type == 'Simple':
            # weights = (target_volume.float().sqrt() + self.eps).reciprocal()
            weights = (target_volume.float()**(1./3.) + self.eps).reciprocal()
            # temp_weights = torch.where(torch.isinf(weights), torch.ones_like(weights), weights)
            # max_weights = temp_weights.max(dim=1, keepdim=True)[0]
            # weights = torch.where(torch.isinf(weights), torch.ones_like(weights)*max_weights, weights)
        if self.weight_type == 'Square':
            weights = (target_volume ** 2 + self.eps).float().reciprocal()
            # weights = 1/(target_volume ** 2+self.eps)
            temp_weights = torch.where(torch.isinf(weights), torch.ones_like(weights), weights)
            max_weights = temp_weights.max(dim=1, keepdim=True)[0]
            weights = torch.where(torch.isinf(weights), torch.ones_like(weights) * max_weights, weights)
        elif self.weight_type == 'Uniform':
            weights = torch.ones(shape[0], shape[1] - int(self.no_bg))
        else:
            ValueError("Class weighting type {} does not exists!".format(self.weight_type))
        weights = weights / weights.max()
        # print(weights)
        weights = weights.to(source.device)

        intersection = (source_flat * target_flat).sum(2)
        scores = (2. * (intersection.float()) + self.eps) / (
                (source_volume.float() + target_volume.float()) + 2 * self.eps)
        # scores = (2. * torch.sum(intersection.float() * weights, 1) + self.eps) / (
        #         weights * torch.sum(source_volume.float() + target_volume.float(), 1) + 2*self.eps)

        return 1 - (weights*scores).sum()/weights.sum()

class LogDiceLossMultiClass(nn.Module):
    """Dice loss from two inputs of segmentation masks(different with between a mask and a probability map)"""

    def __init__(self, n_class=None, weight_type='Simple', eps=1e-10):
        super(LogDiceLossMultiClass, self).__init__()
        self.weight_type = weight_type
        self.n_class = n_class
        self.eps = eps

        # self.source_one_hot = nn.Parameter()
        # self.target_one_hot = nn.Parameter()

    def forward(self, source, target):
        """

        :param source: Tensor of size BxCxDxMxN, One-hot encoding mask /class-wise probability of prediction
        :param target: Tensor, ground truth mask when of size Bx1xDxMxN;
                             One-hot encoding mask /class-wise probability of prediction when of size BxCxDxMxN
        :return:
        """
        assert source.shape[0] == target.shape[0]
        assert source.shape[2:] == target.shape[2:]

        if self.n_class is None:
            self.n_class = max(torch.unique(target).max(), torch.unique(source).max()).long().item() + 1

        shape = list(source.shape)

        # flat the spatial dimensions
        source_flat = source.view(shape[0], shape[1], -1)

        # flat the spatial dimensions and transform it into one-hot coding
        if target.shape[1] == 1:
            target_flat = mask_to_one_hot(target.view(shape[0], 1, -1), self.n_class)
        elif target.shape[1] == shape[1]:
            target_flat = target.view(shape[0], shape[1], -1)
        else:
            ValueError("Incorrect size of target tensor: {}, should be {} or []".format(target.shape, shape,
                                                                                        shape[:1] + [1, ] + shape[2:]))

        # does not consider background
        # source_flat = source_flat[:, 1:, :]
        # target_flat = target_flat[:, 1:, :]

        #
        source_volume = source_flat.sum(2)
        target_volume = target_flat.sum(2)

        if self.weight_type == 'Simple':
            weights = (target_volume.float().sqrt() + self.eps).reciprocal()
            # temp_weights = torch.where(torch.isinf(weights), torch.ones_like(weights), weights)
            # max_weights = temp_weights.max(dim=1, keepdim=True)[0]
            # weights = torch.where(torch.isinf(weights), torch.ones_like(weights)*max_weights, weights)
        if self.weight_type == 'Square':
            weights = (target_volume ** 2 + self.eps).float().reciprocal()
            # weights = 1/(target_volume ** 2+self.eps)
            temp_weights = torch.where(torch.isinf(weights), torch.ones_like(weights), weights)
            max_weights = temp_weights.max(dim=1, keepdim=True)[0]
            weights = torch.where(torch.isinf(weights), torch.ones_like(weights) * max_weights, weights)
        elif self.weight_type == 'Uniform':
            weights = torch.ones(shape[0], shape[1] - 1)
        else:
            ValueError("Class weighting type {} does not exists!".format(self.weight_type))
        weights = weights / weights.max()
        # print(weights)
        weights = weights.to(source.device)

        intersection = (source_flat * target_flat).sum(2)
        scores = (2. * (intersection.float() * weights) + self.eps) / (
                (weights * (source_volume.float() + target_volume.float())) + 2 * self.eps)
        # scores = (2. * torch.sum(intersection.float() * weights, 1) + self.eps) / (
        #         weights * torch.sum(source_volume.float() + target_volume.float(), 1) + 2*self.eps)

        return -torch.log(scores.mean())



class GeneralizedDiceLoss(nn.Module):
    """Dice loss from two inputs of segmentation masks(different with between a mask and a probability map)"""

    def __init__(self, n_class=None, weight_type='Simple', no_bg=False, eps=1e-10):
        super(GeneralizedDiceLoss, self).__init__()
        self.weight_type = weight_type
        self.n_class = n_class
        self.no_bg = no_bg ## do not count background class
        self.eps = eps

        # self.source_one_hot = nn.Parameter()
        # self.target_one_hot = nn.Parameter()

    def forward(self, source, target):
        """

        :param source: Tensor of size BxCxDxMxN, One-hot encoding mask /class-wise probability of prediction
        :param target: Tensor, ground truth mask when of size Bx1xDxMxN;
                             One-hot encoding mask /class-wise probability of prediction when of size BxCxDxMxN
        :return:
        """
        assert source.shape[0] == target.shape[0]
        assert source.shape[2:] == target.shape[2:]

        if self.n_class is None:
            self.n_class = max(torch.unique(target).max(), torch.unique(source).max()).long().item() + 1

        shape = list(source.shape)

        # flat the spatial dimensions
        source_flat = source.view(shape[0], shape[1], -1)

        # flat the spatial dimensions and transform it into one-hot coding
        if target.shape[1] == 1:
            target_flat = mask_to_one_hot(target.view(shape[0], 1, -1), self.n_class)
        elif target.shape[1] == shape[1]:
            target_flat = target.view(shape[0], shape[1], -1)
        else:
            ValueError("Incorrect size of target tensor: {}, should be {} or []".format(target.shape, shape, shape[:1]+[1,]+shape[2:]))

        # does not consider background
        if self.no_bg:
            source_flat = source_flat[:, 1:, :]
            target_flat = target_flat[:, 1:, :]

        source_volume = source_flat.sum(2)
        target_volume = target_flat.sum(2)

        if self.weight_type == 'Simple':
            weights = (target_volume.float()+self.eps).reciprocal()
            temp_weights = torch.where(torch.isinf(weights), torch.ones_like(weights), weights)
            max_weights = temp_weights.max(dim=1, keepdim=True)[0]
            weights = torch.where(torch.isinf(weights), torch.ones_like(weights)*max_weights, weights)
        if self.weight_type == 'Square':
            weights = (target_volume.float()**2 + self.eps).reciprocal()
            # weights = 1/(target_volume ** 2+self.eps)
            temp_weights = torch.where(torch.isinf(weights), torch.ones_like(weights), weights)
            max_weights = temp_weights.max(dim=1, keepdim=True)[0]
            weights = torch.where(torch.isinf(weights), torch.ones_like(weights)*max_weights, weights)
        elif self.weight_type == 'Uniform':
            weights = torch.ones(shape[0], shape[1]-1)
        else:
            ValueError("Class weighting type {} does not exists!".format(self.weight_type))
        weights = weights / weights.max()
        # print(weights)
        weights = weights.to(source.device)

        intersection = (source_flat * target_flat).sum(2)
        scores = (2. * torch.sum(intersection.float() * weights, 1) + self.eps) / (torch.sum(
                weights * (source_volume.float() + target_volume.float()), 1) + 2*self.eps)

        return 1 - scores.mean()


class LogGeneralizedDiceLoss(nn.Module):
    """Dice loss from two inputs of segmentation masks(different with between a mask and a probability map)"""

    def __init__(self, n_class=None, weight_type='Simple', eps=1e-10):
        super(LogGeneralizedDiceLoss, self).__init__()
        self.weight_type = weight_type
        self.n_class = n_class
        self.eps = eps

        # self.source_one_hot = nn.Parameter()
        # self.target_one_hot = nn.Parameter()

    def forward(self, source, target):
        """

        :param source: Tensor of size BxCxDxMxN, One-hot encoding mask /class-wise probability of prediction
        :param target: Tensor, ground truth mask when of size Bx1xDxMxN;
                             One-hot encoding mask /class-wise probability of prediction when of size BxCxDxMxN
        :return:
        """
        assert source.shape[0] == target.shape[0]
        assert source.shape[2:] == target.shape[2:]

        if self.n_class is None:
            self.n_class = max(torch.unique(target).max(), torch.unique(source).max()).long().item() + 1

        shape = list(source.shape)

        # flat the spatial dimensions
        source_flat = source.view(shape[0], shape[1], -1)

        # flat the spatial dimensions and transform it into one-hot coding
        if target.shape[1] == 1:
            target_flat = mask_to_one_hot(target.view(shape[0], 1, -1), self.n_class)
        elif target.shape[1] == shape[1]:
            target_flat = target.view(shape[0], shape[1], -1)
        else:
            ValueError("Incorrect size of target tensor: {}, should be {} or []".format(target.shape, shape, shape[:1]+[1,]+shape[2:]))

        # does not consider background
        # source_flat = source_flat[:, 1:, :]
        # target_flat = target_flat[:, 1:, :]

        #
        source_volume = source_flat.sum(2)
        target_volume = target_flat.sum(2)

        if self.weight_type == 'Simple':
            weights = (target_volume.float()+self.eps).reciprocal()
            temp_weights = torch.where(torch.isinf(weights), torch.ones_like(weights), weights)
            max_weights = temp_weights.max(dim=1, keepdim=True)[0]
            weights = torch.where(torch.isinf(weights), torch.ones_like(weights)*max_weights, weights)
        if self.weight_type == 'Square':
            weights = (target_volume**2 + self.eps).float().reciprocal()
            # weights = 1/(target_volume ** 2+self.eps)
            temp_weights = torch.where(torch.isinf(weights), torch.ones_like(weights), weights)
            max_weights = temp_weights.max(dim=1, keepdim=True)[0]
            weights = torch.where(torch.isinf(weights), torch.ones_like(weights)*max_weights, weights)
        elif self.weight_type == 'Uniform':
            weights = torch.ones(shape[0], shape[1]-int(self.no_bg))
        else:
            ValueError("Class weighting type {} does not exists!".format(self.weight_type))
        weights = weights / weights.max()
        # print(weights)
        weights = weights.to(source.device)

        intersection = (source_flat * target_flat).sum(2)
        scores = (2. * torch.sum(intersection.float() * weights, 1) + self.eps) / (torch.sum(
            weights * (source_volume.float() + target_volume.float()), 1) + 2 * self.eps)
        return -torch.log(scores.mean())


class GeneralizedDiceLoss4Organs(nn.Module):
    def __init__(self, organIDs=[1], size_average=True):
        super(GeneralizedDiceLoss4Organs, self).__init__()
        self.organIDs = organIDs
        self.size_average = size_average

    def forward(self, inputs, targets, save=True):
        """
            Args:
                inputs:(n, c, h, w, d)
                targets:(n, h, w, d): 0,1,...,C-1
        """
        assert not targets.requires_grad
        assert inputs.dim() == 5, inputs.shape
        assert targets.dim() == 5, targets.shape
        assert inputs.size(0) == targets.size(0), "{0} vs {1} ".format(inputs.size(0), targets.size(0))
        assert inputs.size(2) == targets.size(2), "{0} vs {1} ".format(inputs.size(2), targets.size(1))
        assert inputs.size(3) == targets.size(3), "{0} vs {1} ".format(inputs.size(3), targets.size(2))
        assert inputs.size(4) == targets.size(4), "{0} vs {1} ".format(inputs.size(4), targets.size(3))

        eps = Variable(torch.cuda.FloatTensor(1).fill_(0.000001))
        one = Variable(torch.cuda.FloatTensor(1).fill_(1.0))
        two = Variable(torch.cuda.FloatTensor(1).fill_(2.0))

        inputSZ = inputs.size()  # it should be sth like NxCxHxW

        inputs = F.softmax(inputs, dim=1)

        numOfCategories = inputSZ[1]
        assert numOfCategories == len(self.organIDs), 'organ weights is not matched with organs (bg should be included)'
        ####### Convert categorical to one-hot format

        results_one_hot = inputs

        # target1 = Variable(torch.unsqueeze(targets.data, 1))  # Nx1xHxW

        targets_one_hot = Variable(torch.cuda.FloatTensor(inputSZ).zero_())  # NxCxHxW
        targets_one_hot.scatter_(1, targets, 1)  # scatter along the 'numOfDims' dimension

        ###### Now the prediction and target has become one-hot format
        ###### Compute the dice for each organ
        out = Variable(torch.cuda.FloatTensor(1).zero_(), requires_grad=True)
        #     intersect = Variable(torch.cuda.FloatTensor([1]).zero_(), requires_grad = True)
        #     union = Variable(torch.cuda.FloatTensor([1]).zero_(), requires_grad = True)

        intersect = Variable(torch.cuda.FloatTensor(1).fill_(0.0))
        union = Variable(torch.cuda.FloatTensor(1).fill_(0.0))

        for organID in range(0, numOfCategories):

            target = targets_one_hot[:, organID, ...].contiguous().view(-1, 1).squeeze(1)  # for 2D or 3D
            result = results_one_hot[:, organID, ...].contiguous().view(-1, 1).squeeze(1)  # for 2D or 3D
            #             print 'unique(target): ',unique(target),' unique(result): ',unique(result)
            if torch.sum(target).cpu().item() == 0:
                organWeight = Variable(torch.cuda.FloatTensor(1).fill_(
                    0.0))  # this is necessary, otherwise, union can be too big due to too big organ weight if some organ doesnot appear
            else:
                organWeight = 1 / ((torch.sum(target)) ** 2 + eps)
            #             print 'sum: %d'%torch.sum(target),' organWeight: %f'%organWeight
            #         intersect = torch.dot(result, target)
            intersect_vec = result * target
            intersect = intersect + organWeight * torch.sum(intersect_vec)
            #         print type(intersect)
            # binary values so sum the same as sum of squares
            result_sum = torch.sum(result)
            #         print type(result_sum)
            target_sum = torch.sum(target)
            union = union + organWeight * (result_sum + target_sum) + (two * eps)

            # the target volume can be empty - so we still want to
            # end up with a score of 1 if the result is 0/0

        IoU = intersect / union
        #             out = torch.add(out, IoU.data*2)
        out = one - two * IoU

        #     print type(out)
        return out

"""
Image similarity loss
"""


class NormalizedCrossCorrelationLoss(nn.Module):
    """
    The ncc loss: 1- NCC
    """

    def __init__(self):
        super(NormalizedCrossCorrelationLoss, self).__init__()

    def forward(self, input: torch.tensor, target: torch.tensor):
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)
        input_minus_mean = input - torch.mean(input, 1, keepdim=True)
        target_minus_mean = target - torch.mean(target, 1, keepdim=True)
        nccSqr = (input_minus_mean * target_minus_mean).mean(1) / (
            torch.sqrt((input_minus_mean**2).mean(1)) * torch.sqrt((target_minus_mean**2).mean(1)))
        nccSqr = nccSqr.mean()
        return 1 - nccSqr


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, input: torch.tensor, target: torch.tensor):
        return ((input - target) ** 2).mean()


class LNCCLoss(nn.Module):
    def initialize(self, kernel_sz=[9, 9, 9], voxel_weights=None):
        pass

    def __stepup(self, img_sz, use_multi_scale=True):
        max_scale = min(img_sz)
        if use_multi_scale:
            if max_scale > 128:
                self.scale = [int(max_scale / 16), int(max_scale / 8), int(max_scale / 4)]
                self.scale_weight = [0.1, 0.3, 0.6]
                self.dilation = [2, 2, 2]


            elif max_scale > 64:
                self.scale = [int(max_scale / 4), int(max_scale / 2)]
                self.scale_weight = [0.3, 0.7]
                self.dilation = [2, 2]
            else:
                self.scale = [int(max_scale / 2)]
                self.scale_weight = [1.0]
                self.dilation = [1]
        else:
            self.scale_weight = [int(max_scale / 4)]
            self.scale_weight = [1.0]
        self.num_scale = len(self.scale)
        self.kernel_sz = [[scale for _ in range(3)] for scale in self.scale]
        self.step = [[max(int((ksz + 1) / 4), 1) for ksz in self.kernel_sz[scale_id]] for scale_id in
                     range(self.num_scale)]
        self.filter = [torch.ones([1, 1] + self.kernel_sz[scale_id]).cuda() for scale_id in range(self.num_scale)]

        self.conv = F.conv3d

    def forward(self, input, target, inst_weights=None, train=None):
        self.__stepup(img_sz=list(input.shape[2:]))
        input_2 = input ** 2
        target_2 = target ** 2
        input_target = input * target
        lncc_total = 0.
        for scale_id in range(self.num_scale):
            input_local_sum = self.conv(input, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                        stride=self.step[scale_id]).view(input.shape[0], -1)
            target_local_sum = self.conv(target, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                         stride=self.step[scale_id]).view(input.shape[0],
                                                                          -1)
            input_2_local_sum = self.conv(input_2, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                          stride=self.step[scale_id]).view(input.shape[0],
                                                                           -1)
            target_2_local_sum = self.conv(target_2, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                           stride=self.step[scale_id]).view(
                input.shape[0], -1)
            input_target_local_sum = self.conv(input_target, self.filter[scale_id], padding=0,
                                               dilation=self.dilation[scale_id], stride=self.step[scale_id]).view(
                input.shape[0], -1)

            input_local_sum = input_local_sum.contiguous()
            target_local_sum = target_local_sum.contiguous()
            input_2_local_sum = input_2_local_sum.contiguous()
            target_2_local_sum = target_2_local_sum.contiguous()
            input_target_local_sum = input_target_local_sum.contiguous()

            numel = float(np.array(self.kernel_sz[scale_id]).prod())

            input_local_mean = input_local_sum / numel
            target_local_mean = target_local_sum / numel

            cross = input_target_local_sum - target_local_mean * input_local_sum - \
                    input_local_mean * target_local_sum + target_local_mean * input_local_mean * numel
            input_local_var = input_2_local_sum - 2 * input_local_mean * input_local_sum + input_local_mean ** 2 * numel
            target_local_var = target_2_local_sum - 2 * target_local_mean * target_local_sum + target_local_mean ** 2 * numel

            lncc = cross * cross / (input_local_var * target_local_var + 1e-5)
            lncc = 1 - lncc.mean()
            lncc_total += lncc * self.scale_weight[scale_id]

        return lncc_total


class VoxelMorphLNCC(nn.Module):
    def __init__(self, filter_size=9, eps=1e-6):
        super(VoxelMorphLNCC, self).__init__()
        self.filter_size = filter_size
        self.win_numel = self.filter_size ** 3
        self.filter = nn.Parameter(torch.ones(1, 1, filter_size, filter_size, filter_size))
        self.eps = eps

    def forward(self, I, J):
        I_square = I ** 2
        J_square = J ** 2
        I_J = I * J

        I_local_sum = F.conv3d(I, self.filter, padding=0)
        J_local_sum = F.conv3d(J, self.filter, padding=0)
        I_square_local_sum = F.conv3d(I_square, self.filter, padding=0)
        J_square_local_sum = F.conv3d(J_square, self.filter, padding=0)
        I_J_local_sum = F.conv3d(I_J, self.filter, padding=0)

        I_local_mean = I_local_sum / self.win_numel
        J_local_mean = J_local_sum / self.win_numel

        cross = I_J_local_sum - I_local_mean * J_local_sum - J_local_mean * I_local_sum + I_local_mean * J_local_mean * self.win_numel
        I_var = I_square_local_sum - 2 * I_local_mean * I_local_sum + I_local_mean ** 2 * self.win_numel
        J_var = J_square_local_sum - 2 * J_local_mean * J_local_sum + J_local_mean ** 2 * self.win_numel

        cc = (cross ** 2)/ (I_var * J_var + self.eps)

        return 1 - cc.mean()

# def cc3D(win=[9, 9, 9], voxel_weights=None):
#     def loss(I, J):
#         I2 = I * I
#         J2 = J * J
#         IJ = I * J
#
#         filt = tf.ones([win[0], win[1], win[2], 1, 1])
#
#         I_sum = tf.nn.conv3d(I, filt, [1, 1, 1, 1, 1], "SAME")
#         J_sum = tf.nn.conv3d(J, filt, [1, 1, 1, 1, 1], "SAME")
#         I2_sum = tf.nn.conv3d(I2, filt, [1, 1, 1, 1, 1], "SAME")
#         J2_sum = tf.nn.conv3d(J2, filt, [1, 1, 1, 1, 1], "SAME")
#         IJ_sum = tf.nn.conv3d(IJ, filt, [1, 1, 1, 1, 1], "SAME")
#
#         win_size = win[0] * win[1] * win[2]
#         u_I = I_sum / win_size
#         u_J = J_sum / win_size
#
#         cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
#         I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
#         J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
#
#         cc = cross * cross / (I_var * J_var + 1e-5)
#
#         # if(voxel_weights is not None):
#         #	cc = cc * voxel_weights
#
#         return -1.0 * tf.reduce_mean(cc)
#
#     return loss
"""
deformation field regularization loss
"""


class gradientLoss(nn.Module):
    """
    regularization loss of the spatial gradient of a 3d deformation field
    """

    def __init__(self, norm='L2'):
        super(gradientLoss, self).__init__()
        self.norm = norm

    def forward(self, input):
        """
        :param norm: 'L1' or 'L2'
        :param input: Nx3xDxHxW
        :return:
        """
        dx = torch.abs(input[:, :, 2:, :, :] + input[:, :, :-2, :, :] - 2 * input[:, :, 1:-1, :, :])
        dy = torch.abs(input[:, :, :, 2:, :] + input[:, :, :, :-2, :] - 2 * input[:, :, :, 1:-1, :])
        dz = torch.abs(input[:, :, :, :, 2:] + input[:, :, :, :, :-2] - 2 * input[:, :, :, :, 1:-1])
        if self.norm == 'L2':
            dx = dx ** 2
            dy = dy ** 2
            dz = dz ** 2
        d = (dx.mean() + dy.mean() + dz.mean()) / 3.0
        return d


class L2Loss(nn.Module):

    def forward(self, input):
        return (input ** 2).mean()


loss_dict = {
    'ncc': NormalizedCrossCorrelationLoss,
    'lncc': VoxelMorphLNCC,
    'mse': nn.MSELoss,
    'gradient': gradientLoss,
    'diceLabel': DiceLossOnLabel,
    'logGenDice': LogGeneralizedDiceLoss,
    'genDice': GeneralizedDiceLoss,
    'dice': DiceLossMultiClass,
    'L2': L2Loss,
    'focal':FocalLoss
}


def get_loss_function(loss_name):
    if loss_name in get_available_losses():
        return loss_dict[loss_name]
    else:
        KeyError("Network {} is not avaiable!\n Choose from: {}".format(loss_name, get_available_losses()))


def get_available_losses():
    return loss_dict.keys()


if __name__ == '__main__':
    input = torch.randn(2, 1, 160, 160, 160)
    target = torch.randn(2, 1, 160, 160, 160)
    criterion2 = NormalizedCrossCorrelationLoss()
    loss2 = criterion2(input, input)
