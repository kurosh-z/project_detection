import os
import numpy as np


supported = [
    "type",
    "batch_normalize",
    "filters",
    "size",
    "stride",
    "pad",
    "activation",
    "layers",
    "groups",
    "from",
    "mask",
    "anchors",
    "classes",
    "num",
    "jitter",
    "ignore_thresh",
    "truth_thresh",
    "random",
    "stride_x",
    "stride_y",
    "weights_type",
    "weights_normalization",
    "scale_x_y",
    "beta_nms",
    "nms_kind",
    "iou_loss",
    "iou_normalizer",
    "cls_normalizer",
    "iou_thresh",
]


# def parse_model_config(path):
#     if not path.endswith(".cfg"):  # add .cfg suffix if omitted
#         path += ".cfg"
#     if not os.path.exists(path) and os.path.exists("cfg" + os.sep + path):  # add cfg/ prefix if omitted
#         path = "cfg" + os.sep + path

#     with open(path, "r") as f:
#         lines = f.read().split("\n")
#     lines = [x for x in lines if x and not x.startswith("#")]
#     lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
#     module_defs = []  # module definitions
#     for line in lines:
#         if line.startswith("["):  # This marks the start of a new block
#             module_defs.append({})
#             module_defs[-1]["type"] = line[1:-1].rstrip()
#             if module_defs[-1]["type"] == "convolutional":
#                 module_defs[-1]["batch_normalize"] = 0  # pre-populate with zeros (may be overwritten later)
#         else:
#             key, val = line.split("=")
#             key = key.rstrip()

#             if key == "anchors":  # return nparray
#                 module_defs[-1][key] = np.array([float(x) for x in val.split(",")]).reshape((-1, 2))  # np anchors
#             elif (key in ["from", "layers", "mask"]) or (key == "size" and "," in val):  # return array
#                 module_defs[-1][key] = [int(x) for x in val.split(",")]
#             else:
#                 val = val.strip()
#                 if val.isnumeric():  # return int or float
#                     module_defs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
#                 else:
#                     module_defs[-1][key] = val  # return string

#     # Check all fields are supported

#     f = []  # fields
#     for x in module_defs[1:]:
#         [f.append(k) for k in x if k not in f]
#     u = [x for x in f if x not in supported]  # unsupported fields
#     assert not any(u), "Unsupported fields %s in %s" % (u, path)

#     return module_defs


def parse_model_config(path):
    """
    Parses the yolo-v3 layer configuration file and returns module definitions

    Args
        path: configuration file path

    Returns
        Module definition as a list

    """
    file = open(path, "r")
    lines = file.read().split("\n")
    lines = [x for x in lines if x and not x.startswith("#")]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith("["):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]["type"] = line[1:-1].rstrip()
            if module_defs[-1]["type"] == "convolutional":
                module_defs[-1]["batch_normalize"] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
    return module_defs


if __name__ == "__main__":
    path = "/Users/kurosh/Documents/DEV/Python/preject-detection/models/yolov3.cfg"
    modules = parse_model_config(path)
