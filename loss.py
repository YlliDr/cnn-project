import torch
import torch.nn as nn


# binary cross entropy works well for mask prediction
bce_loss = nn.BCELoss()


def soft_iou_score(prediction, target, smooth=1e-6):
    # flatten everything so we compare all pixels directly
    prediction = prediction.view(-1)
    target = target.view(-1)

    # pixels where prediction and target overlap
    intersection = (prediction * target).sum()

    # total area covered by prediction or target
    union = prediction.sum() + target.sum() - intersection

    # smooth avoids division by zero
    iou = (intersection + smooth) / (union + smooth)

    return iou


def sod_loss(prediction, target):
    # normal pixel-by-pixel loss
    bce = bce_loss(prediction, target)

    # iou part helps the model focus on mask shape
    iou = soft_iou_score(prediction, target)

    # required loss from the project description
    loss = bce + 0.5 * (1 - iou)

    return loss