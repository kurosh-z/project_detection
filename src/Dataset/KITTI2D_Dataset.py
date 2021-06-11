from __future__ import print_function, division
import os
import torch
import random
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from ..utils.show_image import show_image
from ..utils.gUtils import mkdir_nested
from ..utils.torchUtils import worker_seed_set
from skimage.util import img_as_ubyte, img_as_float
from .transforms import MResize
from src.Dataset.transforms import DEFAULT_TRANSFORMS, AUGMENTATION_TRANSFORMS, RelativeLabels
from ..utils.augUtil import draw_rect

get_relative_labels = RelativeLabels()


def create_TrainValidate_data(path, ratio=0.8, seed=None):
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
        bb = np.array([b.tolist() for b in bb]).reshape((-1, 4))
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
        bb = np.array([b.tolist() for b in bb]).reshape((-1, 4))
        if bb.shape[0] == 0 or classes.shape[0] == 0:
            continue
        k2dValidate.write(validNameX)
        k2dValidate.write("\n")
        labels = np.concatenate((classes, bb), axis=1)
        np.save(os.path.join(validate_path, validNameY), labels, allow_pickle=True)
    k2dValidate.close()


def create_Test_data(path):

    test_path = os.path.join(os.path.join(path, "test"))
    if not os.path.exists(test_path):
        mkdir_nested(test_path)

    x_test = np.load(os.path.join(path, "x_test.npy"), allow_pickle=True)
    k2dTest = open("/Users/kurosh/Documents/DEV/python/project_detection/data/k2dTest.txt", "w")

    for index in range(x_test.shape[0]):
        testNameX = "k2d_TestX_{:02d}.npy".format(index)
        np.save(os.path.join(test_path, testNameX), x_test[index, ...], allow_pickle=True)

        k2dTest.write(testNameX)
        k2dTest.write("\n")

    k2dTest.close()


def create_train_data_loader(data_path, batch_size, img_size, n_cpu):
    """Returns DataLoader for training.

    img_path:(str) Path to file containing all paths to training images.
    batch_size:(int) Size of each image batch
    n_cpu:(int) Number of cpu threads to use during batch generation
    """
    # TODO: add parameter for multiscale training: Scale images to different sizes

    dataset = KITTI2D(path=data_path, mode="Train", image_size=img_size, transform=AUGMENTATION_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        shuffle=True,
        worker_init_fn=worker_seed_set,
    )
    return dataloader


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class KITTI2D_Test(Dataset):
    """KITTI2D test dataset."""

    def __init__(self, path, image_size=416, transform=None):
        """ """
        self.path = path
        self.image_size = image_size
        self.transform = transform
        self._load_file_names()

    def __len__(self):
        return len(self.filenames)

    def _load_file_names(self):
        # _path = os.path.join(self.path, "k2dTest.txt")
        _path = os.path.join(self.path, "k2dTrain.txt")
        names_txt = open(_path, "r")
        self.filenames = names_txt.readlines()

    def _get_file_path(self, idx):
        name = self.filenames[idx % len(self.filenames)].rstrip()
        # return os.path.join(self.path, "test/" + name)
        return os.path.join(self.path, "train/" + name)

    def _load_data(self, idx):
        path = self._get_file_path(idx)
        print(f"{idx}: {path}")
        self.x = np.load(path, allow_pickle=True)
        return path, self.x

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, raw_image = self._load_data(idx)

        # Label Placeholder just for making transformation work!
        labels = np.zeros((1, 5))

        resizer = MResize(self.image_size)
        image, labels = resizer(raw_image, labels)

        if self.transform:
            image, labels = self.transform((image, labels))

        return img_path, image


class KITTI2D(Dataset):
    """KITTI2D train and validate dataset."""

    def __init__(
        self,
        path,
        mode="Train",
        image_size=416,
        max_objects=50,
        transform=None,
        multiscale=True,
    ):
        """ """
        assert (
            mode == "Train" or mode == "Validate" or mode == "Test"
        ), "expected Train or Validate as mode got: {}".format(mode)
        image_size[0] if isinstance(image_size, tuple) else image_size
        self.path = path
        self.mode = mode
        self.image_size = image_size
        self.min_size = self.image_size - 3 * 32
        self.max_size = self.image_size + 3 * 32
        self.transform = transform
        self.multiscale = multiscale
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
        # converto dtype('uint8') for augmentation
        self.x = img_as_ubyte(self.x)
        self.y = np.load(paths["y"], allow_pickle=True)

    def __len__(self):
        return len(self.filenames)

    def collate_fn(self, batch):
        images, labels = list(zip(*batch))
        # Remove empty placeholder targets
        imgs = torch.stack([img for img in images])

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.image_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.image_size) for img in imgs])

        # Add sample index to targets
        for idx, boxes in enumerate(labels):
            boxes[:, 0] = idx
        labels = torch.cat(labels, 0)

        return imgs, labels

    def _prepare(self, image, labels):

        # resizer = tr.Resize(self.image_size)

        # resizedImg, resizedBBoxes = resizer(image, labels)
        # calculate yolov variables for bbx:

        _labels = np.zeros((len(self.y), 5))
        for idx, bbox in enumerate(labels):
            xc = (bbox[2] + bbox[4]) / 2.0
            yc = (bbox[1] + bbox[3]) / 2.0
            w = bbox[4] - bbox[2]
            h = bbox[3] - bbox[1]
            _labels[idx, ...] = np.array([bbox[0], xc, yc, w, h])

        image, labels = get_relative_labels((image, _labels))

        return image, _labels

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

    create_TrainValidate_data(path)

    # trainset = KITTI2D(path, mode="Validate")
    # validateset = KITTI2D(path, mode="Validate")
    # testdataset = KITTI2D_Test(path)

    # train_dataloader = DataLoader(dataset=trainset, batch_size=2, shuffle=False, collate_fn=trainset.collate_fn)
    # # train_features, train_labels = next(iter(train_dataloader))
    # for idx, (images, labels) in enumerate(validateset):
    # traindataset = create_train_data_loader(path, 1, 1)
    # traindataset = KITTI2D(path=path, mode="Train", transform=AUGMENTATION_TRANSFORMS)

    # for idx, (image, labels) in enumerate(traindataset):
    #     #     #     # show_image((images, labels))
    #     print(idx)
    #     fig, ax = plt.subplots(1, 1)
    #     ax.imshow(draw_rect(img_as_float(image), labels[:, 1:5], rectype="xywh"))
    #     plt.show()
