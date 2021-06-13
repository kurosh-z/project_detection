#! /usr/bin/env python3

from __future__ import division

import os, sys
import tqdm
import random
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from src.Model.load_model import load_model
from src.Dataset.KITTI2D_Dataset import dataloader_factory
from src.utils.torchUtils import rescale_boxes, non_max_suppression, to_cpu


DEFAULT_CONF = {
    "model": "configs/yolov3-custom.cfg",
    "weights": "checkpoints/yolov3_ckpt_300.pth",
    "test_path": "data",  # root directory where the test numpy files are
    "classes_names": ["Car"],
    "output": "output",
    "batch_size": int(1),
    "img_size": 416,  # input image size of yolo model
    "n_cpu": int(4),
    "iou_thres": 0.5,  # IOU threshold for evalutation
    "conf_thres": 0.5,  # Object confidence threshold
    "nms_thres": 0.5,  # IOU threshold for non-maximum suppression
    "save_images": True,
    "mode": "test-test",
    "max_num_of_imgs": None,
}


def draw_testing(pred, img, ax):
    # Colormap for plotted bounding boxes
    cmap = {0: "r", "Pedestrian": "g", "Cyclist": "b", "Van": "yellow", "Truck": "black"}
    # we deleted labels other than cars

    boxes = pred["boxes"]
    classes = pred["classes"]

    # Plot image
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"prediction test")

    # Plot colored bounding boxes
    for bbox, cls in zip(boxes, classes):
        try:
            cl = cmap[cls]
        except:
            cl = cmap["Misc"]
        # Add rectangle (x,y), width, heigth
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]
        rect = patches.Rectangle((bbox[1], bbox[0]), width, height, linewidth=1, edgecolor=cl, facecolor="none")
        ax.add_patch(rect)


def _store_prediction_results(
    model_path,
    weights_path,
    test_path,
    classes,
    output_path,
    mode,
    batch_size=4,
    img_size=416,
    n_cpu=4,
    conf_thres=0.5,
    nms_thres=0.5,
    save_images=False,
    max_num_of_imgs=None,
):
    """Detects test data and save results"""
    dataloader = dataloader_factory(
        data_path=test_path,
        batch_size=batch_size,
        img_size=img_size,
        n_cpu=n_cpu,
        mode=mode,
        max_num_of_imgs=max_num_of_imgs,
    )
    model = load_model(model_path, weights_path)
    img_detections, imgs = _predict_labels(model, dataloader, output_path, img_size, conf_thres, nms_thres)
    # Iterate through images and save results
    prediction_results = []
    for (image_path, detections) in zip(imgs, img_detections):
        pred_dict = get_prediction_result(image_path, detections, img_size, output_path, classes, save_images)
        prediction_results.append(pred_dict)

    results_name = os.path.join(output_path, "prediction_results.npy")
    prediction_results = np.array(prediction_results)
    np.save(results_name, prediction_results, allow_pickle=True)

    return prediction_results


def _predict_labels(model, dataloader, output_path, img_size, conf_thres, nms_thres):
    """[summary]

    Args:
        model ([type]): [description]
        dataloader ([type]): [description]
        output_path ([type]): [description]
        img_size ([type]): [description]
        conf_thres ([type]): [description]
        nms_thres ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Create output directory, if missing
    os.makedirs(output_path, exist_ok=True)

    model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    img_detections = []  # detections for each image

    imgs = []  # image paths list

    for (img_paths, input_imgs) in tqdm.tqdm(dataloader, desc="Detecting"):
        # for (img_paths, input_imgs) in dataloader:
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # Store image and detections
        img_detections.extend(detections)
        imgs.extend(img_paths)
    return img_detections, imgs


def get_prediction_result(image_path, detections, img_size, output_path, classes, save_images):

    """ """

    img = np.load(image_path)
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.imshow(img)

    # Rescale boxes to original image
    detections = rescale_boxes(detections, img_size, img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    # Bounding-box colors
    cmap = plt.get_cmap("Set1")
    colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
    bbox_colors = random.sample(colors, n_cls_preds)

    predict_dict = {
        "boxes": np.array([]),
        "classes": np.array([]),
        "scores": np.array([]),
    }  # Storing detection results for submission

    for idx, (x1, y1, x2, y2, conf, cls_pred) in enumerate(detections):

        print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

        box_w = x2 - x1
        box_h = y2 - y1
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0

        box_pos_x = (x_center - box_w / 2.0).item()
        box_pos_y = (y_center - box_h / 2.0).item()
        box_height = (box_pos_y + box_h).item()
        box_width = (box_pos_x + box_w).item()

        boxes = np.array([box_pos_y, box_pos_x, box_height, box_width])

        if idx == 0:
            predict_dict["boxes"] = boxes.reshape((1, -1))
            predict_dict["classes"] = cls_pred.detach().cpu().numpy().ravel()
            predict_dict["scores"] = conf.detach().cpu().numpy().ravel()
        else:
            predict_dict["boxes"] = np.vstack((predict_dict["boxes"], boxes))
            predict_dict["classes"] = np.concatenate(
                (predict_dict["classes"], cls_pred.detach().cpu().numpy().ravel()), axis=None
            )
            predict_dict["scores"] = np.concatenate(
                (predict_dict["scores"], conf.detach().cpu().numpy().ravel()), axis=None
            )

        if save_images:
            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            ax.text(
                x1,
                y1,
                s=classes[int(cls_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )
    # shape = predict_dict["boxes"].shape

    if save_images:
        # Save generated image with detections
        filename = os.path.basename(image_path).split(".")[0]
        output_path = os.path.join(output_path, f"{filename}.png")
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        # draw_testing(predict_dict, img, ax2)
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
        # plt.show()
        plt.close()

    return predict_dict


def detectME(**CONF):

    for conf_key, value in DEFAULT_CONF.items():
        if conf_key in CONF:
            print("\n", conf_key, "->", CONF[conf_key])
        else:
            CONF[conf_key] = value
            print("\n", f"{conf_key} set to default value -> {value}")

    prediction_results = _store_prediction_results(
        model_path=CONF["model"],
        weights_path=CONF["weights"],
        test_path=CONF["test_path"],
        classes=CONF["classes_names"],
        output_path=CONF["output"],
        batch_size=CONF["batch_size"],
        img_size=CONF["img_size"],
        n_cpu=CONF["n_cpu"],
        conf_thres=CONF["conf_thres"],
        nms_thres=CONF["nms_thres"],
        save_images=CONF["save_images"],
        mode=CONF["mode"],
        max_num_of_imgs=CONF["max_num_of_imgs"],
    )

    return prediction_results


if __name__ == "__main__":
    curr_path = os.getcwd()
    sys.path.append(curr_path)

    MY_CONF = {
        "model": "configs/yolov3-custom.cfg",
        "weights": "checkpoints/yolov3_ckpt_300.pth",
        "test_path": "data",  # root directory where the test numpy files are
        "classes_names": ["Car"],
        "output": "output",
        "batch_size": int(1),
        "img_size": 416,  # input image size of yolo model
        "n_cpu": int(4),
        "iou_thres": 0.5,  # IOU threshold for evalutation
        "conf_thres": 0.5,  # Object confidence threshold
        "nms_thres": 0.5,  # IOU threshold for non-maximum suppression
        "save_images": True,
        "mode": "test-train",
        "max_num_of_imgs": 1,
    }

    detectME(**MY_CONF)
