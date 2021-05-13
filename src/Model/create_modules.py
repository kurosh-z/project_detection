from __future__ import division

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from .DetectionLayer import DetectionLayer
from .DummyLayer import DummyLayer
from .YOLOLayer import YOLOLayer


# def create_modules(module_defs):
#     """
#     Constructs module list of layer blocks from module configuration in module_defs

#     Args
#         module_defs: list defining the modules/ building blocks

#     Returns
#         hyperparams
#         module list

#     """
#     hyperparams = module_defs.pop(0)
#     output_filters = [int(hyperparams["channels"])]
#     module_list = nn.ModuleList()
#     for i, module_def in enumerate(module_defs):
#         modules = nn.Sequential()

#         if module_def["type"] == "convolutional":
#             bn = int(module_def["batch_normalize"])
#             filters = int(module_def["filters"])
#             kernel_size = int(module_def["size"])
#             pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0
#             modules.add_module(
#                 "conv_%d" % i,
#                 nn.Conv2d(
#                     in_channels=output_filters[-1],
#                     out_channels=filters,
#                     kernel_size=kernel_size,
#                     stride=int(module_def["stride"]),
#                     padding=pad,
#                     bias=not bn,
#                 ),
#             )
#             if bn:
#                 modules.add_module("batch_norm_%d" % i, nn.BatchNorm2d(filters))
#             if module_def["activation"] == "leaky":
#                 modules.add_module("leaky_%d" % i, nn.LeakyReLU(0.1))

#         elif module_def["type"] == "maxpool":
#             kernel_size = int(module_def["size"])
#             stride = int(module_def["stride"])
#             if kernel_size == 2 and stride == 1:
#                 padding = nn.ZeroPad2d((0, 1, 0, 1))
#                 modules.add_module("_debug_padding_%d" % i, padding)
#             maxpool = nn.MaxPool2d(
#                 kernel_size=int(module_def["size"]),
#                 stride=int(module_def["stride"]),
#                 padding=int((kernel_size - 1) // 2),
#             )
#             modules.add_module("maxpool_%d" % i, maxpool)

#         elif module_def["type"] == "upsample":
#             upsample = nn.Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
#             modules.add_module("upsample_%d" % i, upsample)

#         elif module_def["type"] == "route":
#             layers = [int(x) for x in module_def["layers"].split(",")]
#             # filters = sum([output_filters[layer_i] for layer_i in layers])
#             filters = 0
#             for layer_i in layers:
#                 if layer_i > 0:
#                     filters += output_filters[layer_i + 1]
#                 else:
#                     filters += output_filters[layer_i]
#             modules.add_module("route_%d" % i, DummyLayer())

#         elif module_def["type"] == "shortcut":
#             filters = output_filters[int(module_def["from"])]
#             modules.add_module("shortcut_%d" % i, DummyLayer())

#         elif module_def["type"] == "yolo":
#             anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
#             # Extract anchors
#             anchors = [int(x) for x in module_def["anchors"].split(",")]
#             anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
#             anchors = [anchors[i] for i in anchor_idxs]
#             num_classes = int(module_def["classes"])
#             img_height = int(hyperparams["height"])
#             # Define detection layer
#             yolo_layer = YOLOLayer(anchors, num_classes, img_height)
#             modules.add_module("yolo_%d" % i, yolo_layer)
#         # Register module list and number of output filters
#         module_list.append(modules)
#         output_filters.append(filters)

#     return hyperparams, module_list


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0
            modules.add_module(
                "conv_%d" % i,
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module("batch_norm_%d" % i, nn.BatchNorm2d(filters))
            if module_def["activation"] == "leaky":
                modules.add_module("leaky_%d" % i, nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                modules.add_module("_debug_padding_%d" % i, padding)
            maxpool = nn.MaxPool2d(
                kernel_size=int(module_def["size"]),
                stride=int(module_def["stride"]),
                padding=int((kernel_size - 1) // 2),
            )
            modules.add_module("maxpool_%d" % i, maxpool)

        elif module_def["type"] == "upsample":
            upsample = nn.Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module("upsample_%d" % i, upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            # filters = sum([output_filters[layer_i] for layer_i in layers])
            filters = 0
            for layer_i in layers:
                if layer_i > 0:
                    filters += output_filters[layer_i + 1]
                else:
                    filters += output_filters[layer_i]
            modules.add_module("route_%d" % i, DummyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[int(module_def["from"])]
            modules.add_module("shortcut_%d" % i, DummyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_height = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height)
            modules.add_module("yolo_%d" % i, yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list
