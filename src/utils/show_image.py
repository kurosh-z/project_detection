import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import numpy as np


def show_image(sample, title=None):

    cmap = {0: "r", "Pedestrian": "g", "Cyclist": "b", "Van": "yellow", "Truck": "black"}
    # Get boxes & classes for specific image
    image, imLabels = sample

    mode = "numpy"
    if torch.is_tensor(image):
        mode = "tensor"

    if mode == "tensor":
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.squeeze(image.numpy(), axis=0)
        image = image.transpose((1, 2, 0))

        # Plot image
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.imshow(image)
        ax.axis("off")
        if title:
            plt.title(f"Image {title}")

        # Plot colored bounding boxes
        for idx in range(imLabels.shape[1]):
            bbox = imLabels[0, idx, 1::].numpy().squeeze()
            cls = imLabels[0, idx, 0].item()
            bbox = np.squeeze(bbox)
            try:
                cl = cmap[cls]
            except:
                cl = cmap["Misc"]
            # Add rectangle (x,y), width, heigth
            height = bbox[2] - bbox[0]
            width = bbox[3] - bbox[1]
            rect = mpl.patches.Rectangle((bbox[1], bbox[0]), width, height, linewidth=1, edgecolor=cl, facecolor="none")
            ax.add_patch(rect)
    else:
        boxes = imLabels["boxes"]
        classes = imLabels["classes"]

        # Plot image
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.imshow(image)
        ax.axis("off")
        if title:
            plt.title(f"Image {title}")

        # Plot colored bounding boxes
        for bbox, cls in zip(boxes, classes):
            try:
                cl = cmap[cls]
            except:
                cl = cmap["Misc"]
            # Add rectangle (x,y), width, heigth
            height = bbox[2] - bbox[0]
            width = bbox[3] - bbox[1]
            rect = mpl.patches.Rectangle((bbox[1], bbox[0]), width, height, linewidth=1, edgecolor=cl, facecolor="none")
            ax.add_patch(rect)

    plt.show()
