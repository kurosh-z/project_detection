from __future__ import print_function, division
import os
import torch
import torchvision.transforms.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from ..utils.show_image import show_image
from ..utils.gUtils import mkdir_nested
from . import transforms as tr


def create_TrainValidate_Sets(path, ratio=0.8, seed=None):
    x_t = np.load(os.path.join(path, "x_train.npy"))
    y_t = np.load(os.path.join(path, "y_train.npy"), allow_pickle=True)
    size = x_t.shape[0]
    if not seed:
        seed = 0
    # generate indices
    np.random.seed(seed)
    all_indices = np.arange(size)
    shuffledIndices = np.random.permutation(all_indices)
    trainIndices = shuffledIndices[0 : int(ratio * size)]
    Ktrain_x = x_t[trainIndices, ...]
    Ktrain_y = y_t[trainIndices, ...]
    validateIndices = shuffledIndices[int(ratio * size) : :]
    Kvalidate_x = x_t[validateIndices, ...]
    Kvalidate_y = y_t[validateIndices, ...]

    validate_path = os.path.join(os.path.join(path, "validate"))
    train_path = os.path.join(os.path.join(path, "train"))
    if not os.path.exists(validate_path):
        mkdir_nested(validate_path)
    if not os.path.exists(train_path):
        mkdir_nested(train_path)

    np.save(os.path.join(train_path, "kTrainX.npy"), Ktrain_x, allow_pickle=True)
    np.save(os.path.join(train_path, "kTrainY.npy"), Ktrain_y, allow_pickle=True)
    np.save(os.path.join(validate_path, "kValidX.npy"), Kvalidate_x, allow_pickle=True)
    np.save(os.path.join(validate_path, "kValidY.npy"), Kvalidate_y, allow_pickle=True)


def create_Debugging_Sets(path, size, ratio=0.8, seed=None):
    x_t = np.load(os.path.join(path, "x_train.npy"))
    y_t = np.load(os.path.join(path, "y_train.npy"), allow_pickle=True)

    if not seed:
        seed = 0
    # generate indices
    np.random.seed(seed)
    all_indices = np.arange(x_t.shape[0])
    shuffledIndices = np.random.permutation(all_indices)[0:size]
    trainIndices = shuffledIndices[0 : int(ratio * size)]
    Ktrain_x = x_t[trainIndices, ...]
    Ktrain_y = y_t[trainIndices, ...]
    validateIndices = shuffledIndices[int(ratio * size) : :]
    Kvalidate_x = x_t[validateIndices, ...]
    Kvalidate_y = y_t[validateIndices, ...]

    validate_path = os.path.join(os.path.join(path, "debug/validate"))
    train_path = os.path.join(os.path.join(path, "debug/train"))
    if not os.path.exists(validate_path):
        mkdir_nested(validate_path)
    if not os.path.exists(train_path):
        mkdir_nested(train_path)

    np.save(os.path.join(train_path, "kTrainX.npy"), Ktrain_x, allow_pickle=True)
    np.save(os.path.join(train_path, "kTrainY.npy"), Ktrain_y, allow_pickle=True)
    np.save(os.path.join(validate_path, "kValidX.npy"), Kvalidate_x, allow_pickle=True)
    np.save(os.path.join(validate_path, "kValidY.npy"), Kvalidate_y, allow_pickle=True)


class KITTI2D(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path, mode="Train", image_size=(416, 416), max_objects=50, transform=None):
        """ """
        self.path = path
        self.mode = mode
        self.image_size = image_size
        self.transform = transform
        self.max_objects = max_objects
        self.batch_count = 0
        self._load_data()

    def _load_data(self):
        if self.mode == "Train":
            self.x = np.load(os.path.join(self.path, "kTrainX.npy"), allow_pickle=True)
            self.y = np.load(os.path.join(self.path, "kTrainY.npy"), allow_pickle=True)
        if self.mode == "Validate":
            self.x = np.load(os.path.join(self.path, "kValidX.npy"), allow_pickle=True)
            self.y = np.load(os.path.join(self.path, "kValidY.npy"), allow_pickle=True)

    def __len__(self):
        return self.x.shape[0]

    def collate_fn(self, batch):
        images, labels = list(zip(*batch))
        # Remove empty placeholder targets
        imgs = torch.stack([img for img in images])
        
           # Add sample index to targets
        for idx, boxes in enumerate(labels):
            boxes[:, 0] = idx
        labels = torch.cat(labels, 0)

       

        return imgs, labels

    def _prepare(self, image, labels):

        classes = labels["classes"]
        bboxes = labels["boxes"]
        resizer = tr.Resize(self.image_size)

        _bboxes = None
        if len(bboxes):
            _bboxes = bboxes[0]
            _bboxes = _bboxes[np.newaxis, ...]
            for idx in range(1, len(bboxes)):
                _bboxes = np.vstack((_bboxes, bboxes[idx]))

        resizedImg, resizedBBoxes = resizer(image, _bboxes)
        # calculate yolov variables for bbx:
        labels = np.zeros((len(bboxes), 5))
        for idx, bbox in enumerate(resizedBBoxes):
            xc = (bbox[1] + bbox[3]) / 2.0
            yc = (bbox[0] + bbox[2]) / 2.0
            w = bbox[3] - bbox[1]
            h = bbox[2] - bbox[0]
            labels[idx, ...] = np.array([classes[idx], xc, yc, w, h])

        tensorConvertor = tr.ToTensor()

        return tensorConvertor((resizedImg, labels))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        raw_image = self.x[idx, ...]
        raw_labels = self.y[idx]

        image, labels = self._prepare(raw_image, raw_labels)

        return image, labels


if __name__ == "__main__":

    path = "/Users/kurosh/Documents/DEV/python/project_detection/data/debug"
    # create_Debugging_Sets(path, 100)
    # create_TrainValidate_Sets(path, ratio=0.8)
    trainset = KITTI2D(path, mode="Train")
    # validateset = KITTI2D(path, mode="Validate")

    train_dataloader = DataLoader(dataset=trainset, batch_size=2, shuffle=False, collate_fn=trainset.collate_fn)
    # train_features, train_labels = next(iter(train_dataloader))
    for idx, (images, labels) in enumerate(train_dataloader):
        # show_image((images, labels))
        print(idx)

    # pass

    # show_image((train_features, train_labels))

    # plt.imshow(draw_rect(img, bboxes))
    # plt.show()
