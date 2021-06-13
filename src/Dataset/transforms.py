import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from src.utils.torchUtils import xywh2xyxy_np
from src.utils.augUtil import *


class Scale(object):
    """Scales the image
    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    Parameters
    ----------
    scale_x: float
        The factor by which the image is scaled horizontally
    scale_y: float
        The factor by which the image is scaled vertically
    Returns
    -------
    numpy.ndaaray
        Scaled image in the numpy format of shape `HxWxC`
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
    """

    def __init__(self, scale_x=0.2, scale_y=0.2):
        self.scale_x = scale_x
        self.scale_y = scale_y

    def __call__(self, img, bboxes):

        # Chose a random digit to scale by

        img_shape = img.shape

        resize_scale_x = 1 + self.scale_x
        resize_scale_y = 1 + self.scale_y

        img = cv2.resize(img, None, fx=resize_scale_x, fy=resize_scale_y)

        bboxes[:, :4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]

        canvas = np.zeros(img_shape, dtype=np.uint8)

        y_lim = int(min(resize_scale_y, 1) * img_shape[0])
        x_lim = int(min(resize_scale_x, 1) * img_shape[1])

        canvas[:y_lim, :x_lim, :] = img[:y_lim, :x_lim, :]

        img = canvas
        bboxes = clip_box(bboxes, [0, 0, 1 + img_shape[1], img_shape[0]], 0.25)

        return img, bboxes


class Rotate(object):
    """Rotates an image
    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    Parameters
    ----------
    angle: float
        The angle by which the image is to be rotated
    Returns
    -------
    numpy.ndaaray
        Rotated image in the numpy format of shape `HxWxC`
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
    """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, bboxes):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """

        angle = self.angle
        print(self.angle)

        w, h = img.shape[1], img.shape[0]
        cx, cy = w // 2, h // 2

        corners = get_corners(bboxes)

        corners = np.hstack((corners, bboxes[:, 4:]))

        img = rotate_im(img, angle)

        corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)

        new_bbox = get_enclosing_box(corners)

        scale_factor_x = img.shape[1] / w

        scale_factor_y = img.shape[0] / h

        img = cv2.resize(img, (w, h))

        new_bbox[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]

        bboxes = new_bbox

        bboxes = clip_box(bboxes, [0, 0, w, h], 0.25)

        return img, bboxes


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
    def __init__(self):
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


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, data):
        img, boxes = data
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)

        bb_targets = torch.zeros((len(boxes), 6))
        bb_targets[:, 1:] = transforms.ToTensor()(boxes)

        return img, bb_targets


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


class MinimumAug(ImgAug):
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


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


REQUIRED_TRANSFORMS = transforms.Compose(
    [
        # AbsoluteLabels(),
        PadSquare(),
        RelativeLabels(),
        ToTensor(),
    ]
)

TEST_TRANSFORMS = transforms.Compose(
    [
        ToTensor(),
    ]
)


TRAIN_AUGMENTATION = transforms.Compose(
    [
        MinimumAug(),
        PadSquare(),
        RelativeLabels(),
        ToTensor(),
    ]
)
