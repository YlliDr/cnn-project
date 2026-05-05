import torch


def calculate_metrics(prediction, target, threshold=0.5, smooth=1e-6):
    # convert soft predictions into black/white mask
    prediction = (prediction > threshold).float()
    target = (target > threshold).float()

    # correctly predicted foreground pixels
    true_positive = (prediction * target).sum()

    # pixels predicted as foreground but actually background
    false_positive = (prediction * (1 - target)).sum()

    # pixels that should be foreground but model missed them
    false_negative = ((1 - prediction) * target).sum()

    # how many predicted foreground pixels were actually correct
    precision = (true_positive + smooth) / (
        true_positive + false_positive + smooth
    )

    # how much of the real object the model found
    recall = (true_positive + smooth) / (
        true_positive + false_negative + smooth
    )

    # balance between precision and recall
    f1 = 2 * precision * recall / (precision + recall + smooth)

    # overlap between predicted mask and real mask
    intersection = (prediction * target).sum()
    union = prediction.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)

    # average pixel difference between prediction and target
    mae = torch.abs(prediction - target).mean()

    return {
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "mae": mae.item()
    }