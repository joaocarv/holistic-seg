from torch import sigmoid, bincount
import torch.nn as nn
import torch.nn.functional as F
from torch import mean, diag
import argparse
from scipy.spatial.distance import directed_hausdorff

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ---- Loss Functions
class Dice_Loss(nn.Module):
    """
    Taken from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
    """
    def __init__(self, weight=None, size_average=True):
        super(Dice_Loss, self).__init__()

    def forward(self, out, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        out = sigmoid(out)
        # flatten label and prediction tensors
        out = out.view(-1)
        targets = targets.view(-1)

        intersection = (out * targets).sum()
        dice = (2. * intersection + smooth) / (out.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceCE_Loss(nn.Module):
    """
    Taken from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
    """

    def __init__(self, weight=None, size_average=True):
        super(DiceCE_Loss, self).__init__()

    def forward(self, out, targets, smooth=1e-5):


        # Binary Cross entropy loss
        BCE = F.binary_cross_entropy_with_logits(out, targets, reduction='mean')

        out = sigmoid(out)
        # flatten label and prediction tensors
        num = targets.size(0)
        out = out.view(num,-1)
        targets = targets.view(num,-1)


        # Compute dice loss
        intersection = (out * targets)
        dice = (2. * intersection.sum(1) + smooth) / (out.sum(1) + targets.sum(1) + smooth)
        # dice_loss = 1 - dice.sum() / num
        # Dice_BCE = 0.5 * BCE + dice_loss

        dice_loss = dice.sum() / num
        Dice_BCE = 0.5 * BCE - dice_loss

        return Dice_BCE


class CE_Loss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(CE_Loss, self).__init__()

    def forward(self, input,target,n_classes):

        loss = F.cross_entropy(input, target) if n_classes > 1 else \
            F.binary_cross_entropy_with_logits(input, target)

        return loss


# ----- Evaluation Metrics
def _fast_hist(out, target, num_classes):
    num_classes=num_classes+1
    mask = (target >= 0) & (target < num_classes)

    hist = bincount(
        num_classes * target[mask] + out[mask],
        minlength=num_classes ** 2,
    )
    hist = hist.reshape(num_classes, num_classes)
    hist = hist.float()
    return hist

def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return mean(x[x == x])

def jaccard_index(hist, smooth=1):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).
    Args:
        hist: confusion matrix.
    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    A_inter_B = diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + smooth)
    avg_jacc = nanmean(jaccard)
    return avg_jacc

def dice_coefficient(hist, smooth =1):
    """Computes the Sørensen–Dice coefficient, a.k.a the F1 score.
    Args:
        hist: confusion matrix.
    Returns:
        avg_dice: the average per-class dice coefficient.
    """
    A_inter_B = diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B + smooth)  / (A + B + smooth)
    avg_dice = nanmean(dice)
    return avg_dice


def dice_jaccard_sing_class(inputs, target, smooth=1):
    intersection = ((target * inputs).sum() + smooth).float()
    union_sum = (target.sum() + inputs.sum() + smooth).float()
    if target.sum() == 0 and inputs.sum() == 0:
        return (1.0,1.0)

    dice_score = 2.0 * intersection / union_sum
    jaccard_score = intersection / (union_sum - intersection)

    return dice_score, jaccard_score


def hausforff_distance_scipy(out, target):
    out = out.detach().to('cpu').numpy().astype(bool)
    target = target.to('cpu').numpy().astype(bool)

    avg_hd = []
    for i in range(target.shape[0]):

        hd = max(directed_hausdorff(target[i,0, :, :], out[i,0, :, :]),
            directed_hausdorff(out[i, 0,:, :], target[i,0, :, :]))
        avg_hd.append(hd)


    return np.array(avg_hd).mean(), np.array(avg_hd).mean()


def eval_metrics(out, target, num_classes):
    avg_hd_95, avg_hd_100 = hausforff_distance_scipy(out, target)

    if num_classes>1:
        hist = _fast_hist(out, target, num_classes)
        avg_jaccard = jaccard_index(hist)
        avg_dice = dice_coefficient(hist)
    else:
        avg_dice, avg_jaccard = dice_jaccard_sing_class(out, target)

    return {'avg_dice': avg_dice,
        'avg_jaccard': avg_jaccard,
        'avg_hd_95': avg_hd_95,
        'avg_hd_100': avg_hd_100}



