#! /usr/bin/env python3

from __future__ import division

import os, sys
import tqdm
import torch
import torch.optim as optim
from terminaltables import AsciiTable
from torchsummary import summary

from src.Model.load_model import load_model
from src.utils.gUtils import Logger
from src.utils.torchUtils import to_cpu, set_seed
from src.utils.loss import compute_loss
from src.Dataset.KITTI2D_Dataset import dataloader_factory
from src.testME import evaluate

TRAIN_CONFIG = {
    "model": "configs/yolov3-custom.cfg",
    "pretrained_weights": "checkpoints/yolov3_ckpt_300.pth",  # if the training should start with pretrained-weights
    "train_path": "data",
    "valid_path": "data",
    "classes_names": ["Car", "Van", "Truck", "Pedestrian", "Misc"],
    "logdir": "logs",
    "epochs": int(200),
    "epochBaseNum": int(
        0
    ),  # NOTE:set this if you want to continue with pretrained wights but don't want to overwrite previous weights
    "n_cpu": int(4),
    "seed": -1,  # with any number other than -1 it sets the seeds for numpy and troch
    "checkpoint_interval": int(1),
    "iou_thres": 0.5,  # IOU threshold for evalutation
    "conf_thres": 0.1,  # Object confidence threshold
    "nms_thres": 0.5,  # IOU threshold for non-maximum suppression
    "multiscale_training": False,  # NOTE: this option dosent produce good results for now and needs more testing
    "verbose": False,
}


def trainME(CONFIG):
    print(f"Training began with following configs: \n")
    for key, value in CONFIG.items():
        print(key, "->", value)

    if CONFIG["seed"] != -1:
        set_seed(CONFIG["seed"])

    epochBaseNum = CONFIG["epochBaseNum"]
    # set tensorboard logging
    logger = Logger(CONFIG["logdir"])

    # Create directories
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n Creating the Model with pretraind { CONFIG['pretrained_weights']} ...\n")
    model = load_model(CONFIG["model"], CONFIG["pretrained_weights"])

    # Print model:
    if CONFIG["verbose"]:
        summary(model, input_size=(3, model.hyperparams["height"], model.hyperparams["height"]))

    mini_batch_size = model.hyperparams["batch"] // model.hyperparams["subdivisions"]
    # create training and validation dataloaders

    dataloader = dataloader_factory(
        data_path=CONFIG["train_path"],
        batch_size=mini_batch_size,
        img_size=model.hyperparams["height"],
        n_cpu=CONFIG["n_cpu"],
        mode="Train",
        multiscale_training=False,
    )

    # Load validation dataloader
    validation_dataloader = dataloader_factory(
        data_path=CONFIG["valid_path"],
        batch_size=mini_batch_size,
        img_size=model.hyperparams["height"],
        n_cpu=CONFIG["n_cpu"],
        mode="Validate",
    )

    # optimization:
    params = [p for p in model.parameters() if p.requires_grad]
    if model.hyperparams["optimizer"] == None:
        model.hyperparams["optimizer"] = "adam"

    if model.hyperparams["optimizer"] == "adam":
        optimizer = optim.Adam(
            params,
            lr=model.hyperparams["learning_rate"],
            weight_decay=model.hyperparams["decay"],
        )
    elif model.hyperparams["optimizer"] == "sgd":
        optimizer = optim.SGD(
            params,
            lr=model.hyperparams["learning_rate"],
            weight_decay=model.hyperparams["decay"],
            momentum=model.hyperparams["momentum"],
        )
    else:
        raise Exception("\n Expected one of adam or sgd for optimizer got {}".format(model.hyperparams["optimizer"]))

    for epoch in range(CONFIG["epochs"]):

        print("\n---- Training Epoch ----")

        model.train()

        for batch_i, (imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epochBaseNum + epoch}")):
            # for batch_i, (imgs, targets) in enumerate(dataloader):

            batches_done = len(dataloader) * epoch + batch_i
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)
            outputs = model(imgs)
            loss, loss_components = compute_loss(outputs, targets, model)
            loss.backward()

            # optimizer
            if batches_done % model.hyperparams["subdivisions"] == 0:
                # adapting learning rate
                lr = model.hyperparams["learning_rate"]
                if batches_done < model.hyperparams["burn_in"]:
                    # Burn in
                    lr *= batches_done / model.hyperparams["burn_in"]
                else:
                    # else set learning rate got from hyperparams
                    for threshold, value in model.hyperparams["lr_steps"]:
                        if batches_done > threshold:
                            lr *= value
                # Log the learning rate
                logger.scalar_summary("train/learning_rate", lr, batches_done)
                # Set learning rate
                for g in optimizer.param_groups:
                    g["lr"] = lr
                optimizer.step()
                optimizer.zero_grad()

                # set loggings
                print(
                    AsciiTable(
                        [
                            ["Type", "Value"],
                            ["IoU loss", float(loss_components[0])],
                            ["Object loss", float(loss_components[1])],
                            ["Class loss", float(loss_components[2])],
                            ["Loss", float(loss_components[3])],
                            ["Batch loss", to_cpu(loss).item()],
                        ]
                    ).table
                )

            # Tensorboard:
            tensorboard_log = [
                ("train/iou_loss", float(loss_components[0])),
                ("train/obj_loss", float(loss_components[1])),
                ("train/class_loss", float(loss_components[2])),
                ("train/loss", to_cpu(loss).item()),
            ]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)
            model.seen += imgs.size(0)

        # Save model to checkpoint file
        if epoch % CONFIG["checkpoint_interval"] == 0:
            checkpointNum = epochBaseNum + epoch
            checkpoint_path = f"checkpoints/yolov3_ckpt_{checkpointNum}.pth"
            print(f"---- saving checkpoint: '{checkpoint_path}' ----")
            torch.save(model.state_dict(), checkpoint_path)

        print("\n---- Evaluating ----")
        # Evaluate the model on the validation set
        metrics_output = evaluate(
            model,
            validation_dataloader,
            CONFIG["classes_names"],
            img_size=model.hyperparams["height"],
            iou_thres=CONFIG["iou_thres"],
            conf_thres=CONFIG["conf_thres"],
            nms_thres=CONFIG["nms_thres"],
            verbose=CONFIG["verbose"],
        )

        if metrics_output is not None:
            precision, recall, AP, f1, ap_class = metrics_output
            evaluation_metrics = [
                ("validation/precision", precision.mean()),
                ("validation/recall", recall.mean()),
                ("validation/mAP", AP.mean()),
                ("validation/f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)


if __name__ == "__main__":
    curr_path = os.getcwd()
    sys.path.append(curr_path)
    trainME(TRAIN_CONFIG)
