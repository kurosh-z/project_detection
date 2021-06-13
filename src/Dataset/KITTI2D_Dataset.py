from __future__ import print_function, division
import os, sys
import torch
import random
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from src.utils.show_image import show_image
from src.utils.gUtils import mkdir_nested
from src.utils.torchUtils import worker_seed_set
from skimage.util import img_as_ubyte, img_as_float
from src.Dataset.transforms import MResize, resize, REQUIRED_TRANSFORMS, TRAIN_AUGMENTATION, TEST_TRANSFORMS
from src.utils.augUtil import draw_rect


def prepare_TrainValidate_data(inp_path, out_path, ratio=0.8, seed=None):
    x_t = np.load(os.path.join(inp_path, "x_train.npy"))
    y_t = np.load(os.path.join(inp_path, "y_train.npy"), allow_pickle=True)
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

    validate_path = os.path.join(os.path.join(out_path, "validate"))
    train_path = os.path.join(os.path.join(out_path, "train"))
    if not os.path.exists(validate_path):
        mkdir_nested(validate_path)
    if not os.path.exists(train_path):
        mkdir_nested(train_path)

    k2dTrain = open(os.path.join(out_path, "k2dTrain.txt"), "w")
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

    k2dValidate = open(os.path.join(out_path, "k2dValidate.txt"), "w")
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
        k2dValidate.write(validNameX)
        k2dValidate.write("\n")
        labels = np.concatenate((classes, bb), axis=1)
        np.save(os.path.join(validate_path, validNameY), labels, allow_pickle=True)
    k2dValidate.close()


def prepare_test_data(inp_path, out_path):

    test_path = os.path.join(os.path.join(out_path, "test"))
    if not os.path.exists(test_path):
        mkdir_nested(test_path)

    x_test = np.load(os.path.join(inp_path, "x_test.npy"), allow_pickle=True)
    k2dTest = open(os.path.join(out_path, "k2dTest.txt"), "w")

    for index in range(x_test.shape[0]):
        testNameX = "k2d_TestX_{:02d}.npy".format(index)
        np.save(os.path.join(test_path, testNameX), x_test[index, ...], allow_pickle=True)

        k2dTest.write(testNameX)
        k2dTest.write("\n")

    k2dTest.close()


def dataloader_factory(
    data_path, mode="Train", n_cpu=4, batch_size=1, img_size=416, multiscale_training=False, max_num_of_imgs=None
):
    """[factory for dataloader training or validation]
    Args:
        data_path ([str]): path to data directory
        batch_size ([int]): number of batches
        n_cpu ([int]): cpu threads: Defaults to 4.
        img_size (int, optional): Defaults to 416.
        mode (str, optional):[one of Train , Validate, test-test test-train]  Defaults to "Train".
        multiscale_training (bool, optional): [just for training]. Defaults to False.

    Returns:
       [dataloader]
    """
    assert mode in ["Train", "Validate", "test-test", "test-train"]
    if mode in ["test-test", "test-train"]:
        dataset = KITTI2D_Test(
            path=data_path,
            image_size=img_size,
            mode=mode,
            max_num_of_imgs=max_num_of_imgs,
            transform=TEST_TRANSFORMS,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=n_cpu,
            pin_memory=True,
            shuffle=False,
        )
        return dataloader

    transform = REQUIRED_TRANSFORMS if mode == "Validate" else TRAIN_AUGMENTATION

    dataset = KITTI2D(
        path=data_path, image_size=img_size, mode=mode, transform=transform, multiscale=multiscale_training
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        shuffle=True if mode == "Train" else False,
        worker_init_fn=worker_seed_set if mode == "Train" else None,
    )
    return dataloader


class KITTI2D_Test(Dataset):
    """KITTI2D test dataset."""

    def __init__(self, path, image_size=416, mode="test-test", max_num_of_imgs=None, transform=None):
        """ """
        assert mode == "test-test" or mode == "test-train", "expected test-test or test-train as mode got: {}".format(
            mode
        )
        self.path = path
        self.image_size = image_size
        self.mode = mode
        self.transform = transform
        self.max_num_of_imgs = max_num_of_imgs
        self._load_file_names()

    def __len__(self):
        if self.max_num_of_imgs:
            return min(self.max_num_of_imgs, len(self.filenames))
        return len(self.filenames)

    def _load_file_names(self):
        _path = ""
        if self.mode == "test-test":
            _path = os.path.join(self.path, "k2dTest.txt")
        else:
            _path = os.path.join(self.path, "k2dTrain.txt")

        names_txt = open(_path, "r")
        self.filenames = names_txt.readlines()

    def _get_file_path(self, idx):
        name = self.filenames[idx % len(self.filenames)].rstrip()
        if self.mode == "test-test":
            return os.path.join(self.path, "test/" + name)
        else:
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
        multiscale=False,
        max_num_of_imgs=None,
    ):
        """ """
        assert (
            mode == "Train" or mode == "Validate" or mode == "Test"
        ), "expected Train or Validate as mode got: {}".format(mode)
        self.path = path
        self.mode = mode
        self.image_size = image_size
        self.transform = transform
        self.multiscale = multiscale
        self.max_num_of_imgs = max_num_of_imgs
        self.min_size = self.image_size - 2 * 32
        self.max_size = self.image_size + 2 * 32
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
        if self.max_num_of_imgs:
            return min(self.max_num_of_imgs, len(self.filenames))
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

        # image, labels = get_relative_labels((image, _labels))

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
    curr_path = os.getcwd()
    sys.path.append(curr_path)

    path = "data"

    # create_TrainValidate_Sets(path)
    # create_Test_Set(path)

    traindataset = KITTI2D(path=path, mode="Train")

    for idx, (image, labels) in enumerate(traindataset):
        #     #     # show_image((images, labels))
        print(idx)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(draw_rect(img_as_float(image), labels[:, 1:5], rectype="xywh"))
        plt.show()
