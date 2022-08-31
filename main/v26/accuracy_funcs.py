import torch
from v26.ConstantsBuTd import dev


def get_bounding_box(mask) -> list:
    """_summary_

    Args:
        mask (_type_): _description_

    Returns:
        list: list of 4 points of the bounding box
    """
    if len(mask.shape) > 2:
        bool_mask = mask[:, :, 1]
    else:
        bool_mask = mask
    h, w = bool_mask.shape
    rows, cols = bool_mask.nonzero()
    margin = 5
    stx = min(cols)
    sty = min(rows)
    endx = max(cols)
    endy = max(rows)
    stx -= margin
    sty -= margin
    endx += margin
    endy += margin
    stx = max(0, stx)
    sty = max(0, sty)
    endx = min(endx, w)
    endy = min(endy, h)
    return [stx, sty, endx, endy]


def multi_label_accuracy_base(outs, samples, nclasses):
    """
    multi_label_accuracy_base _summary_

    Args:
        outs (_type_): _description_
        samples (`SimpleNamespace`): Represent a list of samples, maintained when outputing the raw data
        nclasses (_type_): _description_

    Returns:
        _type_: _description_
    """
    cur_batch_size = samples.image.shape[0]
    preds = torch.zeros((cur_batch_size, len(nclasses)),
                        dtype=torch.int).to(dev, non_blocking=True)
    # TODO: check if this is correct - the nclasses is the number of classes in the current task
    preds = torch.argmax(input=outs.task, dim=1, keepdim=False)
    label_task = samples.label_task
    # true label task for each sample, has shape (`number of samples`,`label`)
    # TODO - label_task has shape (`batch_size`,1) and preds has shape (`batch_size`,`number_of_classes`)
    task_accuracy = (preds == label_task).float()
    return preds, task_accuracy


def multi_label_accuracy(outs, samples, nclasses):
    """_summary_

    Args:
        outs (_type_): _description_
        samples (_type_): _description_
        nclasses (_type_): _description_

    Returns:
        _type_: _description_
    """    
    preds, task_accuracy = multi_label_accuracy_base(outs, samples, nclasses)
    task_accuracy = task_accuracy.mean(axis=1)  # per single example
    return preds, task_accuracy


def multi_label_accuracy_weighted_loss(outs, samples, nclasses):
    """_summary_

    Args:
        outs (_type_): _description_
        samples (_type_): _description_
        nclasses (_type_): _description_

    Returns:
        _type_: _description_
    """    
    preds, task_accuracy = multi_label_accuracy_base(outs, samples, nclasses)
    loss_weight = samples.loss_weight
    task_accuracy = task_accuracy * loss_weight
    # It was in the original version only regular mean in the second access
    # we want to do a weighted mean
    # we have 10 - batch size here in the current example
    # task_accuracy = torch.bmm(input=task_accuracy, mat2=loss_weight)
    task_accuracy = task_accuracy.sum(axis=1) / loss_weight.sum(
        axis=1)  # per single example
    return preds, task_accuracy
