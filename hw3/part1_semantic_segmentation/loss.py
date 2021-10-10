import torch


def calc_val_data(preds, masks, num_classes):
    preds = torch.argmax(preds, dim=1)

    """
    Scale added to intersection and union, because calc_val_loss have no interfaces,
     to take shape of preds, masks into account for mean_acc normalization.
    """

    scale = torch.tensor(preds.shape[1:3]).prod()
    preds, masks = torch.nn.functional.one_hot(preds, num_classes), torch.nn.functional.one_hot(masks, num_classes)
    intersection = (preds & masks).sum(dim=(1, 2)) / scale # TODO: calc intersection for each class
    union = (preds | masks).sum(dim=(1, 2)) / scale # TODO: calc union for each class
    target = (preds == masks).sum(dim=(1, 2)) / scale  # TODO: calc number of pixels in groundtruth mask per class
    # Output shapes: B x num_classes

    assert isinstance(intersection, torch.Tensor), 'Output should be a tensor'
    assert isinstance(union, torch.Tensor), 'Output should be a tensor'
    assert isinstance(target, torch.Tensor), 'Output should be a tensor'

    assert intersection.shape == union.shape == target.shape, 'Wrong output shape'
    assert union.shape[0] == masks.shape[0] and union.shape[1] == num_classes, 'Wrong output shape'

    return intersection, union, target


def calc_val_loss(intersection, union, target, eps=1e-7):
    mean_iou = (intersection / (union + eps)).mean() # TODO: calc mean class iou
    mean_class_acc = target.mean() # TODO: calc mean class accuracy
    mean_acc = intersection.sum(dim=1).mean() # TODO: calc mean accuracy

    return mean_iou, mean_class_acc, mean_acc