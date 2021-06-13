#! /usr/bin/env python3

from __future__ import division
import tqdm
import numpy as np
from terminaltables import AsciiTable
import torch
from torch.autograd import Variable
from src.utils.torchUtils import (
    ap_per_class,
    get_batch_statistics,
    non_max_suppression,
    to_cpu,
    xywh2xyxy,
)


def print_evaluation_results(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- WARNING: no detections found by model!!! ----")


def evaluate(model, dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose):
    """Model Evaluation, Returns precision, recall, AP, f1, ap_class

    model: Model to evaluate
    dataloader: Dataloader
    class_names: List of class names
    img_size: Size of each image dimension for yolo
    iou_thres: IOU threshold required to qualify as detected
    conf_thres: Object confidence threshold
    nms_thres: IOU threshold for non-maximum suppression
    """
    model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for imgs, targets in tqdm.tqdm(dataloader, desc="Validating"):
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size
        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = to_cpu(model(imgs))
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- WARNING: No detections over the entire validation set!!! ----")
        return None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    print_evaluation_results(metrics_output, class_names, verbose)

    return metrics_output
