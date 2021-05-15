from __future__ import division
import torch
from .Darknet import Darknet
from ..utils import torchUtils


def load_model(model_path, weights_path=None):
    """Loads the yolo model from file.
    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :return: Returns model
    :rtype: Darknet
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select device for inference
    model = Darknet(model_path).to(device)

    model.apply(torchUtils.weights_init_normal)

    # If pretrained weights are specified, start from checkpoint or weight file
    if weights_path:
        if weights_path.endswith(".pth"):
            # Load checkpoint weights
            model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            # Load darknet weights
            model.load_darknet_weights(weights_path)
    return model
