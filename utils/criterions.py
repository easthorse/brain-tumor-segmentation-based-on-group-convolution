import torch.nn.functional as F
import torch
import logging
import torch.nn as nn
import pdb

__all__ = ['sigmoid_dice_loss', 'softmax_dice_loss', 'GeneralizedDiceLoss', 'FocalLoss']

cross_entropy = F.cross_entropy


def dicefocalloss(output, target, alpha=0.25, gamma=2.0, epoch=None, eps=1e-5, weight_type='square'):
    target[target == 4] = 3

    '''a= torch.tensor([ 0, 0.43, 0.142, 0.43]).cuda()
    #output_l = (flatten(output)[1:, ...]) .transpose(0, 1)  # transpose [N,4,H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    output_l = output.view(output.size(0), output.size(1), -1)  # N,C,H,W,D => N,C,H*W*D
    output_l = output_l.transpose(1, 2)  # N,C,H*W*D => N,H*W*D,C
    output_l = output_l.contiguous().view(-1, output_l.size(2))  # N,H*W*D,C => N*H*W*D,C

    target_l = target.view(-1)
    #
    # output_cut = output_l[:,1:]
    # preds_softmax_cut =F.softmax(output_cut, dim=1)
    # preds_softmax= torch.empty(output_l.size()[0], 4)
    # preds_softmax[:,1:]=preds_softmax_cut
    #preds_softmax[:,0] = -1000
     # N*H*W*D

    #preds_softmax = F.softmax(output_l, dim=1)
    #output yijing guiyihua
    #preds_softmax=F.softmax(output_l, dim=1)
    #pdb.set_trace()
    preds_logsoft = torch.log(output_l)

    preds_logsoft = preds_logsoft *a

    preds_softmax = output_l.gather(1, target_l.view(-1, 1))

    # 让计算出的log先乘以alpha 之后·通过labels调出来
    preds_logsoft = preds_logsoft.gather(1, target_l.view(-1, 1))
    loss_f = -torch.mul(torch.pow((1 - preds_softmax), gamma), preds_logsoft)
    target_l = target
    target_l[target !=0] = 1
    number = target.sum()



    loss_f =loss_f.sum()/number


    '''
    target_d = expand_target(target, n_class=output.size()[1])  # [N,H,W,D] -> [N,4,H,W,D]

    output_d = flatten(output)[1:, ...]  # transpose [N,4,H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target_d = flatten(target_d)[1:, ...]  # [class, N*H*W*D]

    target_sum = target_d.sum(-1)  # sub_class_voxels [3,1] -> 3个voxels  [3,n*h*w*d]->[3,1]
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :', weight_type)

    # print(class_weights)
    intersect = (output_d * target_d).sum(-1)  # [3,1]
    intersect_sum = (intersect * class_weights).sum()  # class_weights is bigger,whose number is smaller ,
    denominator = (output_d + target_d).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2 * intersect[0] / (denominator[0] + eps)
    loss2 = 2 * intersect[1] / (denominator[1] + eps)
    loss3 = 2 * intersect[2] / (denominator[2] + eps)
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(loss1.data, loss2.data, loss3.data))

    loss_d = 1 - 2. * intersect_sum / denominator_sum
    loss_f = FocalLoss(output, target, alpha=0.25, gamma=2.0)
    print(loss_f, loss_d)
    return loss_d + loss_f * 20

    # dice  = GeneralizedDiceLoss(output,target,eps=1e-5,weight_type='square')

    # if epoch >2:
    #     print(epoch)
    # else:
    #     return dice


def FocalLoss(output, target, alpha=0.25, gamma=2.0):
    target[target == 4] = 3
    a = torch.tensor([0.001, 0.43, 0.142, 0.43]).cuda()
    output = output.view(output.size(0), output.size(1), -1)  # N,C,H,W,D => N,C,H*W*D
    output = output.transpose(1, 2)  # N,C,H*W*D => N,H*W*D,C
    output = output.contiguous().view(-1, output.size(2))  # N,H*W*D,C => N*H*W*D,C
    target = target.view(-1)  # N*H*W*D
    # todo
    # preds_softmax = F.softmax(output, dim=1)
    # preds_logsoft = torch.log(preds_softmax)
    # preds_softmax = preds_softmax.gather(1, target.view(-1, 1))
    preds_logsoft = torch.log(output)
    preds_softmax = output.gather(1, target.view(-1, 1))

    preds_logsoft = preds_logsoft * a
    # print(labels.view(-1, 1))
    # 让计算出的log先乘以alpha 之后·通过labels调出来
    preds_logsoft = preds_logsoft.gather(1, target.view(-1, 1))
    loss = -torch.mul(torch.pow((1 - preds_softmax), gamma), preds_logsoft)

    # label [4] -> [3]
    # # target = expand_target(target, n_class=output.size()[1]) # [N,H,W,D] -> [N,4,H,W,D]
    # if output.dim() > 2:
    #     output = output.view(output.size(0), output.size(1), -1)  # N,C,H,W,D => N,C,H*W*D
    #     output = output.transpose(1, 2)  # N,C,H*W*D => N,H*W*D,C
    #     output = output.contiguous().view(-1, output.size(2))  # N,H*W*D,C => N*H*W*D,C
    # if target.dim() == 5:
    #     target = target.contiguous().view(target.size(0), target.size(1), -1)
    #     target = target.transpose(1, 2)
    #     target = target.contiguous().view(-1, target.size(2))
    # if target.dim() == 4:
    #     target = target.view(-1) # N*H*W*D
    # # compute the negative likelyhood
    # logpt = -F.cross_entropy(output, target)
    # pt = torch.exp(logpt)
    # # compute the loss
    # loss = -((1 - pt) ** gamma) * logpt
    # # return loss.sum()
    return loss.mean()


def dice(output, target, eps=1e-5):  # soft dice loss
    target = target.float()
    # num = 2*(output*target).sum() + eps
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num / den


def sigmoid_dice_loss(output, target, alpha=1e-5):
    # output: [-1,3,H,W,T]
    # target: [-1,H,W,T] noted that it includes 0,1,2,4 here
    loss1 = dice(output[:, 0, ...], (target == 1).float(), eps=alpha)
    loss2 = dice(output[:, 1, ...], (target == 2).float(), eps=alpha)
    loss3 = dice(output[:, 2, ...], (target == 4).float(), eps=alpha)
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(1 - loss1.data, 1 - loss2.data, 1 - loss3.data))
    return loss1 + loss2 + loss3


def softmax_dice_loss(output, target, eps=1e-5):  #
    # output : [bsize,c,H,W,D]
    # target : [bsize,H,W,D]
    loss1 = dice(output[:, 1, ...], (target == 1).float())
    loss2 = dice(output[:, 2, ...], (target == 2).float())
    loss3 = dice(output[:, 3, ...], (target == 4).float())
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(1 - loss1.data, 1 - loss2.data, 1 - loss3.data))

    return loss1 + loss2 + loss3


# Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
def GeneralizedDiceLoss(output, target, eps=1e-5, weight_type='square'):  # Generalized dice loss
    """
        Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
    """

    # target = target.float()

    if target.dim() == 4:
        target[target == 4] = 3  # label [4] -> [3]
        target = expand_target(target, n_class=output.size()[1])  # [N,H,W,D] -> [N,4,H,W,D]

    output = flatten(output)[1:, ...]  # transpose [N,4,H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:, ...]  # [class, N*H*W*D]
    #pdb.set_trace()
    target_sum = target.sum(-1)  # sub_class_voxels [3,1] -> 3个voxels  [3,n*h*w*d]->[3,1]
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :', weight_type)

    # print(class_weights)
    intersect = (output * target).sum(-1)  # [3,1]
    intersect_sum = (intersect * class_weights).sum()  # class_weights is bigger,whose number is smaller ,
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2 * intersect[0] / (denominator[0] + eps)
    loss2 = 2 * intersect[1] / (denominator[1] + eps)
    loss3 = 2 * intersect[2] / (denominator[2] + eps)
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(loss1.data, loss2.data, loss3.data))

    return 1 - 2. * intersect_sum / denominator_sum


def mixingdiceLoss(output, target, eps=1e-5, weight_type='square', epoch=None):  # Generalized dice loss
    """
        Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
    """

    # target = target.float()
    # print(epoch)
    if target.dim() == 4:
        target[target == 4] = 3  # label [4] -> [3]
        target = expand_target(target, n_class=output.size()[1])  # [N,H,W,D] -> [N,4,H,W,D]

    output = flatten(output)[1:, ...]  # transpose [N,4,H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:, ...]  # [class, N*H*W*D]

    target_sum = target.sum(-1)  # sub_class_voxels [3,1] -> 3个voxels  [3,n*h*w*d]->[3,1]
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :', weight_type)

    # print(class_weights)
    intersect0 = 2 * (output * target).sum(-1)  # [3,1]
    intersect_sum0 = (intersect0 * class_weights).sum()  # class_weights is bigger,whose number is smaller ,
    denominator0 = (output + target).sum(-1)
    denominator_sum0 = (denominator0 * class_weights).sum() + eps

    # todo
    intersect1 = (output * target).sum(-1)
    intersect_sum1 = (intersect1 * class_weights).sum()
    denominator1 = (output + target - (output * target)).sum(-1)
    denominator_sum1 = (denominator1 * class_weights).sum() + eps

    loss1 = (1 - epoch / 500) * (intersect0[0] / (denominator0[0] + eps)) + 2 * epoch / 500 * (
                intersect1[0] / (denominator1[0] + eps))
    loss2 = (1 - epoch / 500) * (intersect0[1] / (denominator0[1] + eps)) + 2 * epoch / 500 * (
                intersect1[1] / (denominator1[1] + eps))
    loss3 = (1 - epoch / 500) * (intersect0[2] / (denominator0[2] + eps)) + 2 * epoch / 500 * (
                intersect1[2] / (denominator1[2] + eps))
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(loss1.data, loss2.data, loss3.data))
    #print(1)
    return (1 - epoch / 500) * (1 - intersect_sum0 / denominator_sum0) + 2 * epoch / 500 * (
                1 - intersect_sum1 / denominator_sum1)

def jaccardLoss(output, target, eps=1e-5, weight_type='square', epoch=None):  # Generalized dice loss
    """
        Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
    """

    # target = target.float()
    # print(epoch)
    if target.dim() == 4:
        target[target == 4] = 3  # label [4] -> [3]
        target = expand_target(target, n_class=output.size()[1])  # [N,H,W,D] -> [N,4,H,W,D]

    output = flatten(output)[1:, ...]  # transpose [N,4,H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:, ...]  # [class, N*H*W*D]

    target_sum = target.sum(-1)  # sub_class_voxels [3,1] -> 3个voxels  [3,n*h*w*d]->[3,1]
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :', weight_type)

    # print(class_weights)
    intersect0 = 2 * (output * target).sum(-1)  # [3,1]
    intersect_sum0 = (intersect0 * class_weights).sum()  # class_weights is bigger,whose number is smaller ,
    denominator0 = (output + target).sum(-1)
    denominator_sum0 = (denominator0 * class_weights).sum() + eps

    # todo
    intersect1 = (output * target).sum(-1)
    intersect_sum1 = (intersect1 * class_weights).sum()
    denominator1 = (output + target - (output * target)).sum(-1)
    denominator_sum1 = (denominator1 * class_weights).sum() + eps

    loss1 = (1 - epoch / 500) * (intersect0[0] / (denominator0[0] + eps)) + 2 * epoch / 500 * (
                intersect1[0] / (denominator1[0] + eps))
    loss2 = (1 - epoch / 500) * (intersect0[1] / (denominator0[1] + eps)) + 2 * epoch / 500 * (
                intersect1[1] / (denominator1[1] + eps))
    loss3 = (1 - epoch / 500) * (intersect0[2] / (denominator0[2] + eps)) + 2 * epoch / 500 * (
                intersect1[2] / (denominator1[2] + eps))
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(loss1.data, loss2.data, loss3.data))
    print(1)
    return (1 - epoch / 500) * (1 - intersect_sum0 / denominator_sum0) + 2 * epoch / 500 * (
                1 - intersect_sum1 / denominator_sum1)

def StepDiceLoss(output, target, eps=1e-5, weight_type='square', epoch=None):  # Generalized dice loss
    """
        Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
    """

    # target = target.float()

    if target.dim() == 4:
        target[target == 4] = 3  # label [4] -> [3]
        target = expand_target(target, n_class=output.size()[1])  # [N,H,W,D] -> [N,4,H,W,D]

    output = flatten(output)[1:, ...]  # transpose [N,4,H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:, ...]  # [class, N*H*W*D]

    target_sum = target.sum(-1)  # sub_class_voxels [3,1] -> 3个voxels  [3,n*h*w*d]->[3,1]
    # todo

    # if weight_type == 'square':
    #     class_weights = 1. / (target_sum * target_sum + eps)
    # elif weight_type == 'identity':
    #     class_weights = 1. / (target_sum + eps)
    # elif weight_type == 'sqrt':
    #     class_weights = 1. / (torch.sqrt(target_sum) + eps)
    # else:
    #     raise ValueError('Check out the weight_type :',weight_type)

    if epoch < 150:
        class_weights = 1. / (target_sum * target_sum * target_sum + eps)
    elif epoch >= 150 and epoch < 300:
        class_weights = 1. / (target_sum * target_sum + eps)
    elif epoch >= 300:
        class_weights = 1. / (target_sum + eps)
    else:
        raise ValueError('Check out the weight_type :', weight_type)

    # print(class_weights)
    intersect = (output * target).sum(-1)  # [3,1]
    intersect_sum = (intersect * class_weights).sum()  # class_weights is bigger,whose number is smaller ,
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2 * intersect[0] / (denominator[0] + eps)
    loss2 = 2 * intersect[1] / (denominator[1] + eps)
    loss3 = 2 * intersect[2] / (denominator[2] + eps)
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(loss1.data, loss2.data, loss3.data))

    return 1 - 2. * intersect_sum / denominator_sum


def expand_target(x, n_class, mode='softmax'):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
        """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        xx[:, 1, :, :, :] = (x == 1)
        xx[:, 2, :, :, :] = (x == 2)
        xx[:, 3, :, :, :] = (x == 3)
    if mode.lower() == 'sigmoid':
        xx[:, 0, :, :, :] = (x == 1)
        xx[:, 1, :, :, :] = (x == 2)
        xx[:, 2, :, :, :] = (x == 3)
    return xx.to(x.device)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)
