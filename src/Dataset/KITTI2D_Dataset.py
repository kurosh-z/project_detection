from __future__ import print_function, division
import os
import torch
from skimage.transform import resize as imageResize
import torchvision.transforms.functional as F
import imgaug as ia
from imgaug import augmenters as iaa
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from ..utils.show_image import show_image
from ..utils.gUtils import mkdir_nested


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
        self.transform = transform
        self.image_size = image_size
        self.max_objects = max_objects
        self.img_props = {"w": 0, "h": 0, "pad": (0, 0), "padded_h": 0, "padded_w": 0}
        self._load_data()

    def _load_data(self):
        if self.mode == "Train":
            self.x = np.load(os.path.join(self.path, "train/kTrainX.npy"), allow_pickle=True)
            self.y = np.load(os.path.join(self.path, "train/kTrainY.npy"), allow_pickle=True)
        if self.mode == "Validate":
            self.x = np.load(os.path.join(self.path, "validate/kValidX.npy"), allow_pickle=True)
            self.y = np.load(os.path.join(self.path, "validate/kValidY.npy"), allow_pickle=True)

    def __len__(self):
        return self.x.shape[0]

    def _pad_resize_image(self, image):
        """
        Pad and resize the image maintaining the aspect ratio of the image

        Args
            image (np array): 3-dimensional np image array
            image_size (tuple): Integer tuple indicating the size of the image

        Returns
            Padded and resized image as numpy array with channels first

        """

        h, w, _ = image.shape
        dim_diff = np.abs(h - w)

        # Upper left padding
        pad1 = dim_diff // 2

        # lower right padding
        pad2 = 0

        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))

        new_img = np.pad(image, pad, "constant", constant_values=128) / 255.0
        padded_h, padded_w, _ = new_img.shape

        new_img = imageResize(new_img, (*self.image_size, 3), mode="reflect")

        # Channels first for torch operations
        new_img = np.transpose(new_img, (2, 0, 1))

        # modify state variables
        self.img_props["h"] = h
        self.img_props["w"] = w
        self.img_props["pad"] = pad
        self.img_props["padded_h"] = padded_h
        self.img_props["padded_w"] = padded_w

        return new_img

    def _perpare_image(self, image):
        if self.mode == "Train":
            augmented_image = self._augment_image(np.asarray(image))
            resized_image = self._pad_resize_image(augmented_image)
            return torch.from_numpy(resized_image)

        resized_image = self._pad_resize_image(np.asarray(image), self.image_size)
        return resized_image

    def _augment_image(self, image):
        """
        Augment a single image.
        Uses https://github.com/aleju/imgaug library
        Included in requirements (Do not attempt to manually install, use pip install - requirements.txt)

        Args
            image (np array): 3-dimensional np image array

        Returns
            Augmented image as 3-dimensional np image array
        """
        # Add label noise
        rand_int = random.randint(-5, 5)
        value = 0 if rand_int < 0 else rand_int

        seq = iaa.Sequential(
            [
                iaa.SomeOf((0, 2)),
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 0.75)),  # emboss images
                iaa.OneOf(
                    [
                        iaa.GaussianBlur((0, 2.0)),  # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(5, 7)),  # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 5)),  # blur image using local medians with kernel sizes between 2 and 7
                    ]
                ),
                iaa.OneOf(
                    [
                        # either change the brightness of the whole image (sometimes
                        # per channel) or change the brightness of subareas
                        iaa.Multiply((0.8, 1.2), per_channel=0.5),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),  # add gaussian noise to images
                    ]
                ),
                iaa.OneOf(
                    [
                        iaa.Dropout(p=0.05, per_channel=True),
                        iaa.Crop(px=(0, value)),  # crop images from each side by 0 to 4px (randomly chosen)
                    ]
                ),
            ]
        )

        return seq.augment_image(np.asarray(image))

    def _prepare_label(self, imagelabels):
        """
        Read the txt file corresponding to the label and output the label tensor following the yolo format [max_objects x 5]

        Args
            index (int): Index

        Returns
            Torch tensor that encodes the labels for the image

        """
        classes = imagelabels["classes"]
        boxes = imagelabels["boxes"]
        labels = []
        for idx, cl in enumerate(classes):
            item = [cl] + boxes[idx].tolist()
            labels.append(item)
        labels = np.array(labels)
        # # Access state variables
        # w, h, pad, padded_h, padded_w = (
        #     self.img_props["w"],
        #     self.img_props["h"],
        #     self.img_props["pad"],
        #     self.img_props["padded_h"],
        #     self.img_props["padded_w"],
        # )

        # if labels.shape[0]:
        #     # Extract coordinates for unpadded + unscaled image
        #     x1 = w * (labels[:, 1] - labels[:, 3] / 2)
        #     y1 = h * (labels[:, 2] - labels[:, 4] / 2)
        #     x2 = w * (labels[:, 1] + labels[:, 3] / 2)
        #     y2 = h * (labels[:, 2] + labels[:, 4] / 2)

        #     # Adjust for added padding
        #     x1 += pad[1][0]
        #     y1 += pad[0][0]
        #     x2 += pad[1][0]
        #     y2 += pad[0][0]

        #     # Calculate ratios from coordinates
        #     labels[:, 1] = ((x1 + x2) / 2) / padded_w
        #     labels[:, 2] = ((y1 + y2) / 2) / padded_h
        #     labels[:, 3] *= w / padded_w
        #     labels[:, 4] *= h / padded_h

        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))

        if labels.shape[0]:
            filled_labels[range(len(labels))[: self.max_objects]] = labels[: self.max_objects]

        filled_labels = torch.from_numpy(filled_labels)

        return filled_labels

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets

        _targets = []
        for i, target in enumerate(targets):
            boxes = [torch.from_numpy(box) for box in target["boxes"]]
            classes = [torch.tensor(cl) for cl in target["classes"]]
            _targets.append({"boxes": boxes, "classes": classes})

        # targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        # for i, boxes in enumerate(targets):
        #     boxes[:, 0] = i
        # targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        # if self.multiscale and self.batch_count % 10 == 0:
        #     self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # # Resize images to input shape
        # imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        # self.batch_count += 1
        return imgs, _targets

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        raw_image = self.x[idx, ...]
        raw_image_labels = self.y[idx]
        images = self._perpare_image(raw_image)
        image_labels = self._prepare_label(raw_image_labels)

        if self.transform:
            images, image_labels = self.transform((images, image_labels))
            return images, image_labels

        return images, image_labels


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        image, imglabel = data

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return torch.from_numpy(image), torch.from_numpy(imglabel)


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")


if __name__ == "__main__":

    path = "/Users/kurosh/Documents/DEV/python/preject-detection/data/debug"
    # create_Debugging_Sets(path, 100)
    # create_TrainValidate_Sets(path, ratio=0.8)
    trainset = KITTI2D(path, mode="Train")
    validateset = KITTI2D(path, mode="Validate")

    train_dataloader = DataLoader(dataset=trainset, batch_size=1, shuffle=False)
    # train_features, train_labels = next(iter(train_dataloader))
    for i, (images, labels) in enumerate(train_dataloader):
        # pass
        show_image((images, labels))

    # pass

    # show_image((train_features, train_labels))
