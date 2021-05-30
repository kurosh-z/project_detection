from __future__ import print_function, division
import os
import torch
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from ..utils.show_image import show_image
from ..utils.gUtils import mkdir_nested
from . import transforms as tr


from ..utils.augUtil import draw_rect


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

    k2dTrain = open("/Users/kurosh/Documents/DEV/python/project_detection/data/k2dTrain.txt", "w")
    for index in range(Ktrain_x.shape[0]):
        trainNameX = "k2d_trainX_{:02d}.npy".format(index)
        np.save(os.path.join(train_path, trainNameX), Ktrain_x[index, ...], allow_pickle=True)
        trainNameY = "k2d_trainY_{:02d}.npy".format(index)
        item = Ktrain_y[index]
        classes = np.array(item["classes"]).reshape((-1, 1))
        bb = item["boxes"]
        bb = np.array([b.tolist() for b in bb])
        if bb.shape[0] == 0 or classes.shape[0] == 0:
            continue

        k2dTrain.write(trainNameX)
        k2dTrain.write("\n")
        labels = np.concatenate((classes, bb), axis=1)
        # fig, ax = plt.subplots(1, 1)
        # ax.imshow(draw_rect(Ktrain_x[index, ...], labels[:, 1:5]))
        # plt.show()
        np.save(os.path.join(train_path, trainNameY), labels, allow_pickle=True)
    k2dTrain.close()

    k2dValidate = open("/Users/kurosh/Documents/DEV/python/project_detection/data/k2dValidate.txt", "w")
    for index in range(Kvalidate_x.shape[0]):
        validNameX = "k2d_validX_{:02d}.npy".format(index)
        np.save(os.path.join(validate_path, validNameX), Kvalidate_x[index, ...], allow_pickle=True)
        validNameY = "k2d_validY_{:02d}.npy".format(index)
        item = Kvalidate_y[index]
        classes = np.array(item["classes"]).reshape((-1, 1))
        bb = item["boxes"]
        bb = np.array([b.tolist() for b in bb])
        if bb.shape[0] == 0 or classes.shape[0] == 0:
            continue
        k2dValidate.write(trainNameX)
        k2dValidate.write("\n")
        labels = np.concatenate((classes, bb), axis=1)
        np.save(os.path.join(validate_path, validNameY), labels, allow_pickle=True)
    k2dValidate.close()


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
        assert mode == "Train" or mode == "Validate", "expected Train or Validate as mode got: {}".format(mode)
        self.path = path
        self.mode = mode
        self.image_size = image_size
        self.transform = transform
        self.max_objects = max_objects
        self.batch_count = 0
        self._load_file_names()

    def _load_file_names(self):
        _path = ""
        if self.mode == "Validate":
            _path = os.path.join(self.path, "k2dValidate.txt")
        if self.mode == "Train":
            _path = os.path.join(self.path, "k2dTrain.txt")

        names_txt = open(_path, "r")
        self.filenames = names_txt.readlines()

    def _get_file_path(self, idx):
        nameX = self.filenames[idx % len(self.filenames)].rstrip()
        exploded = nameX.split("_")
        nameY = exploded[0] + "_" + exploded[1].replace("X", "Y") + "_" + exploded[2]
        if self.mode == "Train":
            return {"x": os.path.join(self.path, "train/" + nameX), "y": os.path.join(self.path, "train/" + nameY)}
        if self.mode == "Validate":
            return {
                "x": os.path.join(self.path, "validate/" + nameX),
                "y": os.path.join(self.path, "validate/" + nameY),
            }

    def _load_data(self, idx):
        paths = self._get_file_path(idx)
        self.x = np.load(paths["x"], allow_pickle=True)
        self.y = np.load(paths["y"], allow_pickle=True)

    def __len__(self):
        return len(self.filenames)

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

        resizer = tr.Resize(self.image_size)

        resizedImg, resizedBBoxes = resizer(image, labels)
        # calculate yolov variables for bbx:
        labels = np.zeros((len(self.y), 5))
        for idx, bbox in enumerate(resizedBBoxes):
            xc = (bbox[2] + bbox[4]) / 2.0
            yc = (bbox[1] + bbox[3]) / 2.0
            w = bbox[4] - bbox[2]
            h = bbox[3] - bbox[1]
            labels[idx, ...] = np.array([bbox[0], xc, yc, w, h])

        return resizedImg, labels

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        self._load_data(idx)
        raw_image = self.x
        raw_labels = self.y

        image, labels = self._prepare(raw_image, raw_labels)

        if self.transform:
            image, labels = self.transform((image, labels))

        return image, labels


if __name__ == "__main__":

    path = "/Users/kurosh/Documents/DEV/python/project_detection/data"

    # create_TrainValidate_Sets(path)
    # create_Debugging_Sets(path, 100)
    # create_TrainValidate_Sets(path, ratio=0.8)
    trainset = KITTI2D(path, mode="Train")
    # # validateset = KITTI2D(path, mode="Validate")

    # train_dataloader = DataLoader(dataset=trainset, batch_size=2, shuffle=False, collate_fn=trainset.collate_fn)
    # # train_features, train_labels = next(iter(train_dataloader))
    for idx, (images, labels) in enumerate(trainset):

        #     # show_image((images, labels))
        print(idx)

    # pass

    # show_image((train_features, train_labels))

    # plt.imshow(draw_rect(img, bboxes))
    # plt.show()
