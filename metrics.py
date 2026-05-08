import torch


def calculate_metrics(prediction, target, threshold=0.5, smooth=1e-6):
    # keep original soft prediction for MAE
    soft_prediction = prediction.clone()

    # convert soft predictions into binary masks
    prediction = (prediction > threshold).float()
    target = (target > threshold).float()

    # true positives
    true_positive = (prediction * target).sum()

    # false positives
    false_positive = (prediction * (1 - target)).sum()

    # false negatives
    false_negative = ((1 - prediction) * target).sum()

    # precision
    precision = (true_positive + smooth) / (
        true_positive + false_positive + smooth
    )

    # recall
    recall = (true_positive + smooth) / (
        true_positive + false_negative + smooth
    )

    # f1 score
    f1 = 2 * precision * recall / (
        precision + recall + smooth
    )

    # intersection over union
    intersection = (prediction * target).sum()

    union = prediction.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (
        union + smooth
    )

    # MAE should use soft predictions, not thresholded ones
    mae = torch.abs(soft_prediction - target).mean()

    return {
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "mae": mae.item()
    }