import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from ..utils.torchUtils import xywh2xyxy_np


class ImgAug(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, data):
        # Unpack data
        img, boxes = data

        # Convert xywh to xyxy
        boxes = np.array(boxes)
        boxes[:, 1:] = xywh2xyxy_np(boxes[:, 1:])

        # Convert bounding boxes to imgaug
        bounding_boxes = BoundingBoxesOnImage([BoundingBox(*box[1:], label=box[0]) for box in boxes], shape=img.shape)

        # Apply augmentations
        img, bounding_boxes = self.augmentations(image=img, bounding_boxes=bounding_boxes)

        # Clip out of image boxes
        bounding_boxes = bounding_boxes.clip_out_of_image()

        # Convert bounding boxes back to numpy
        boxes = np.zeros((len(bounding_boxes), 5))
        for box_idx, box in enumerate(bounding_boxes):
            # Extract coordinates for unpadded + unscaled image
            x1 = box.x1
            y1 = box.y1
            x2 = box.x2
            y2 = box.y2

            # Returns (x, y, w, h)
            boxes[box_idx, 0] = box.label
            boxes[box_idx, 1] = (x1 + x2) / 2
            boxes[box_idx, 2] = (y1 + y2) / 2
            boxes[box_idx, 3] = x2 - x1
            boxes[box_idx, 4] = y2 - y1

        return img, boxes


class RelativeLabels(object):
    def __init__(
        self,
    ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] /= w
        boxes[:, [2, 4]] /= h
        return img, boxes


class AbsoluteLabels(object):
    def __init__(
        self,
    ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] *= w
        boxes[:, [2, 4]] *= h
        return img, boxes


class PadSquare(ImgAug):
    def __init__(
        self,
    ):
        self.augmentations = iaa.Sequential([iaa.PadToAspectRatio(1.0, position="center-center").to_deterministic()])


# class ToTensor(object):
#     def __init__(
#         self,
#     ):
#         pass

#     def __call__(self, data):
#         img, boxes = data
#         # Extract image as PyTorch tensor
#         img = transforms.ToTensor()(img)

#         bb_targets = torch.zeros((len(boxes), 6))
#         bb_targets[:, 1:] = transforms.ToTensor()(boxes)

#         return img, bb_targets


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, data):
        image, labels = data

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return torch.from_numpy(image), torch.from_numpy(labels)


class MResize(object):
    """Resize the image in accordance to `image_letter_box` function in darknet

    The aspect ratio is maintained. The longer side is resized to the input
    size of the network, while the remaining space on the shorter side is filled
    with black color. **This should be the last transform**


    Parameters
    ----------
    inp_dim : tuple(int)
        tuple containing the size to which the image will be resized.

    Returns
    -------

    numpy.ndaaray
        Sheared image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Resized bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, inp_dim):
        self.inp_dim = inp_dim

    def __call__(self, img, bboxes):
        img_w, img_h = img.shape[1], img.shape[0]
        w = h = self.inp_dim
        new_w = int(img_w * min(w / img_w, h / img_h))
        new_h = int(img_h * min(w / img_w, h / img_h))
        resized_image = cv2.resize(img, (new_w, new_h))
        padded_image = np.full((w, h, 3), 0.0).astype(np.float32)
        padd_h = h - new_h
        padd_w = w - new_w
        padded_image[padd_h // 2 : padd_h // 2 + new_h, padd_w // 2 : padd_w // 2 + new_w, :] = resized_image
        # show_image((img, {"classes": [0 for i in range(len(bboxes))], "boxes": bboxes}))
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(draw_rect(img, bboxes[:, 1:5]))

        scaleY = new_h / img_h
        scaleX = new_w / img_w
        bboxes[:, 1:5] *= np.array([scaleY, scaleX, scaleY, scaleX])
        bboxes[:, 1:5] += np.array([padd_h / 2.0, padd_w / 2.0, padd_h / 2.0, padd_w / 2.0])

        # ax[1].imshow(draw_rect(padded_image,  bboxes[:, 1:5]))
        # plt.show()

        return padded_image, bboxes


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        img, boxes = data
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        return img, boxes


class DefaultAug(ImgAug):
    def __init__(
        self,
    ):
        self.augmentations = iaa.Sequential(
            [
                iaa.Sharpen((0.0, 0.1)),
                iaa.Affine(rotate=(-0, 0), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
                iaa.AddToBrightness((-60, 40)),
                iaa.AddToHue((-10, 10)),
                iaa.Fliplr(0.5),
            ]
        )


class StrongAug(ImgAug):
    def __init__(
        self,
    ):
        self.augmentations = iaa.Sequential(
            [
                iaa.Dropout([0.0, 0.01]),
                iaa.Sharpen((0.0, 0.1)),
                iaa.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
                iaa.AddToBrightness((-60, 40)),
                iaa.AddToHue((-20, 20)),
                iaa.Fliplr(0.5),
            ]
        )


DEFAULT_TRANSFORMS = transforms.Compose(
    [
        AbsoluteLabels(),
        PadSquare(),
        RelativeLabels(),
        ToTensor(),
    ]
)


AUGMENTATION_TRANSFORMS = transforms.Compose(
    [
        AbsoluteLabels(),
        DefaultAug(),
        PadSquare(),
        RelativeLabels(),
        ToTensor(),
    ]
)
