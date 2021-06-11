#! /usr/bin/env python3

from __future__ import division

import os, sys
import argparse
import tqdm
import random
import numpy as np

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from .Model.load_model import load_model
from .Dataset.KITTI2D_Dataset import KITTI2D_Test
from .Dataset.transforms import DEFAULT_TRANSFORMS
from .utils.torchUtils import load_classes, rescale_boxes, non_max_suppression, to_cpu

# from pytorchyolo.utils.datasets import ImageFolder
# from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def detect_directory(
    model_path,
    weights_path,
    img_path,
    classes,
    output_path,
    batch_size=8,
    img_size=416,
    n_cpu=4,
    conf_thres=0.5,
    nms_thres=0.5,
):
    """Detects objects on all images in specified directory and saves output images with drawn detections.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to directory with images to inference
    :type img_path: str
    :param classes: List of class names
    :type classes: [str]
    :param output_path: Path to output directory
    :type output_path: str
    :param batch_size: Size of each image batch, defaults to 8
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    """
    dataloader = _create_data_loader(img_path, batch_size, img_size, n_cpu)
    model = load_model(model_path, weights_path)
    img_detections, imgs = detect(model, dataloader, output_path, img_size, conf_thres, nms_thres)
    _draw_and_save_output_images(img_detections, imgs, img_size, output_path, classes)


def detect_image(model, image, img_size=416, conf_thres=0.5, nms_thres=0.5):
    """Inferences one image with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param image: Image to inference
    :type image: nd.array
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: Detections on image with each detection in the format: [x1, y1, x2, y2, confidence, class]
    :rtype: nd.array
    """
    model.eval()  # Set model to evaluation mode

    # Configure input
    input_img = transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)])((image, np.zeros((1, 5))))[0].unsqueeze(0)

    if torch.cuda.is_available():
        input_img = input_img.to("cuda")

    # Get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        detections = rescale_boxes(detections[0], img_size, image.shape[:2])
    return to_cpu(detections).numpy()


def detect(model, dataloader, output_path, img_size, conf_thres, nms_thres):
    """Inferences images with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images to inference
    :type dataloader: DataLoader
    :param output_path: Path to output directory
    :type output_path: str
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: List of detections. The coordinates are given for the padded image that is provided by the dataloader.
        Use `utils.rescale_boxes` to transform them into the desired input image coordinate system before its transformed by the dataloader),
        List of input image paths
    :rtype: [Tensor], [str]
    """
    # Create output directory, if missing
    os.makedirs(output_path, exist_ok=True)

    model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    img_detections = []  # Stores detections for each image index

    imgs = []  # Stores image paths

    # for (img_paths, input_imgs) in tqdm.tqdm(dataloader, desc="Detecting"):
    for (img_paths, input_imgs) in dataloader:
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


def _draw_and_save_output_images(img_detections, imgs, img_size, output_path, classes):
    """Draws detections in output images and stores them.

    :param img_detections: List of detections
    :type img_detections: [Tensor]
    :param imgs: List of paths to image files
    :type imgs: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param output_path: Path of output directory
    :type output_path: str
    :param classes: List of class names
    :type classes: [str]
    """

    # Iterate through images and save plot of detections
    prediction_results = []
    for (image_path, detections) in zip(imgs, img_detections):
        print(f"Image {image_path}:")
        # _draw_and_save_output_image(image_path, detections, img_size, output_path, classes)
        pred_dict = get_prediction_result(image_path, detections, img_size, output_path, classes)
        prediction_results.append(pred_dict)

    results_name = os.path.join(output_path, "prediction_results.npy")
    np.save(results_name, np.array(prediction_results), allow_pickle=True)


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


def _draw_and_save_output_image(image_path, detections, img_size, output_path, classes):
    """Draws detections in output image and stores this.

    :param image_path: Path to input image
    :type image_path: str
    :param detections: List of detections on image
    :type detections: [Tensor]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param output_path: Path of output directory
    :type output_path: str
    :param classes: List of class names
    :type classes: [str]
    """
    # Create plot
    # img = np.array(Image.open(image_path))
    img = np.load(image_path)
    plt.figure()
    fig, axis = plt.subplots(1, 2)
    ax = axis[0]
    ax2 = axis[1]
    ax.imshow(img)
    # Rescale boxes to original image
    detections = rescale_boxes(detections, img_size, img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
    bbox_colors = random.sample(colors, n_cls_preds)
    predict_dict = {
        "boxes": None,
        "classes": None,
        "scores": None,
    }  # Storing detection results for submission
    predictions_results = []
    for idx, (x1, y1, x2, y2, conf, cls_pred) in enumerate(detections):

        yld, xld, yru, xru = y2, x1, y1, x2
        boxes = np.array([yld, xld, yru, xru])
        if idx == 0:
            predict_dict["boxes"] = boxes
            predict_dict["classes"] = cls_pred.numpy().ravel()
            predict_dict["scores"] = conf.numpy().ravel()
        else:
            predict_dict["boxes"] = np.vstack((predict_dict["boxes"], boxes))
            predict_dict["classes"] = np.concatenate((predict_dict["classes"], cls_pred.numpy().ravel()), axis=None)
            predict_dict["scores"] = np.concatenate((predict_dict["scores"], conf.numpy().ravel()), axis=None)

        print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

        box_w = x2 - x1
        box_h = y2 - y1

        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)
        # Add label
        plt.text(
            x1, y1, s=classes[int(cls_pred)], color="white", verticalalignment="top", bbox={"color": color, "pad": 0}
        )

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    filename = os.path.basename(image_path).split(".")[0]
    output_path = os.path.join(output_path, f"{filename}.png")
    draw_testing(predict_dict, img, ax2)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    plt.show()
    plt.close()


def get_prediction_result(image_path, detections, img_size, output_path, classes):
    """ """
    # Create plot
    # img = np.array(Image.open(image_path))
    img = np.load(image_path)

    fig, axis = plt.subplots(1, 2, figsize=(18, 6))
    ax = axis[0]
    ax2 = axis[1]
    ax.imshow(img)
    ax.set_title("ground truth")

    # Rescale boxes to original image
    detections = rescale_boxes(detections, img_size, img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
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

        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)
        # Add label
        plt.text(
            x1, y1, s=classes[int(cls_pred)], color="white", verticalalignment="top", bbox={"color": color, "pad": 0}
        )
    # shape = predict_dict["boxes"].shape

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    filename = os.path.basename(image_path).split(".")[0]
    output_path = os.path.join(output_path, f"{filename}.png")
    draw_testing(predict_dict, img, ax2)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    plt.show()
    plt.close()

    return predict_dict


def _create_data_loader(img_path, batch_size, img_size, n_cpu):
    """Creates a DataLoader for inferencing.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = KITTI2D_Test(img_path, transform=DEFAULT_TRANSFORMS)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu, pin_memory=True)
    return dataloader


def run():
    parser = argparse.ArgumentParser(description="Detect objects on images.")
    parser.add_argument(
        "-m", "--model", type=str, default="configs/yolov3-custom.cfg", help="Path to model definition file (.cfg)"
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="checkpoints/yolov3_ckpt_300.pth",
        help="Path to weights or checkpoint file (.weights or .pth)",
    )
    parser.add_argument("-i", "--images", type=str, default="data", help="Path to directory with images to inference")
    parser.add_argument(
        "-c", "--classes", type=str, default="data/kitti2d.names", help="Path to classes label file (.names)"
    )
    parser.add_argument("-o", "--output", type=str, default="output", help="Path to output directory")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Size of each image batch")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=4, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    args = parser.parse_args()
    print(f"Detect started with arguments: {args}")

    # Extract class names from file
    classes = load_classes(args.classes)  # List of class names

    detect_directory(
        args.model,
        args.weights,
        args.images,
        classes,
        args.output,
        batch_size=args.batch_size,
        img_size=args.img_size,
        n_cpu=args.n_cpu,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
    )


if __name__ == "__main__":
    curr_path = os.getcwd()
    sys.path.append(curr_path)
    run()
