import argparse
import json
import os
from typing import DefaultDict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BrainSegmentationDataset, DistanceMapDataset 
from logger import Logger
from loss import DiceLoss, DistanceMapLoss, WeightedBCELoss
from transform import transforms
from unet import UNet
from utils import log_images, dsc


def main(args):
    makedirs(args)
    snapshotargs(args)
    device = torch.device(
        "cpu" if not torch.cuda.is_available() else args.device)

    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    unet = UNet(in_channels=BrainSegmentationDataset.in_channels,
                out_channels=BrainSegmentationDataset.out_channels)
    if args.load_weights:
        state_dict = torch.load(args.load_weights, map_location=device)
        unet.load_state_dict(state_dict)
    unet.to(device)

    if args.use_distance_map:
        loss_fn = DistanceMapLoss()
    elif args.use_bce_loss:
        loss_fn = WeightedBCELoss(pos_weight=args.bce_pos_weight)
    else:
        loss_fn = DiceLoss()
    dsc_fn = DiceLoss()
    best_validation_dsc = 0.0
    best_epoch = 0

    optimizer = optim.Adam(unet.parameters(), lr=args.lr)

    logger = Logger(args.logs)
    loss_train = []
    loss_valid = []
    dsc_train = []
    dsc_valid = []

    step = 0

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()

            validation_pred = []
            validation_true = []

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                if args.use_distance_map:
                    x, y_true, y_map = data
                    x, y_true, y_map = x.to(device), y_true.to(device), y_map.to(device)
                else:
                    x, y_true = data
                    x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)

                    if args.use_distance_map:
                        loss = loss_fn(y_pred, y_true, y_map)
                    else:
                        loss = loss_fn(y_pred, y_true)
                    
                    if phase == "valid":
                        loss_valid.append(loss.item())
                        dsc_valid.append(dsc_fn(y_pred, y_true).item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend(
                            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        )
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend(
                            [y_true_np[s] for s in range(y_true_np.shape[0])]
                        )
                        # if (epoch % args.vis_freq == 0) or (epoch == args.epochs - 1):
                        #     if i * args.batch_size < args.vis_images:
                        #         tag = "image/{}".format(i)
                        #         num_images = args.vis_images - i * args.batch_size
                        #         logger.image_list_summary(
                        #             tag,
                        #             log_images(x, y_true, y_pred)[:num_images],
                        #             step,
                        #         )

                    if phase == "train":
                        with torch.set_grad_enabled(False):
                            dsc_train.append(dsc_fn(y_pred, y_true).item())
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0:
                    log_loss_summary(logger, loss_train, step)
                    log_loss_summary(logger, dsc_train, step, prefix="dsc_train_")
                    loss_train = []
                    dsc_train = []

            if phase == "valid":
                log_loss_summary(logger, loss_valid, step, prefix="val_")
                log_loss_summary(logger, dsc_valid, step, prefix='dsc_val_')
                mean_dsc = np.mean(
                    dsc_per_volume(
                        validation_pred,
                        validation_true,
                        loader_valid.dataset.patient_slice_index,
                    )
                )
                logger.scalar_summary("val_dsc", mean_dsc, step)
                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    torch.save(unet.state_dict(), os.path.join(
                        args.weights, "unet.pt"))
                    best_epoch = epoch
                loss_valid = []
                dsc_valid = []

    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))
    print(f'Best model from epoch {best_epoch}')


def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid


def datasets(args):
    if args.use_distance_map:
        train = DistanceMapDataset(
            images_dir=args.train_data,
            subset="train",
            image_size=args.image_size,
            transform=transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=0.5),
            exponent=args.distance_map_exp
        )
        valid = DistanceMapDataset(
            images_dir=args.val_data,
            subset="validation",
            image_size=args.image_size,
            random_sampling=False,
            exponent=args.distance_map_exp
        )
    else:
        train = BrainSegmentationDataset(
            images_dir=args.train_data,
            subset="train",
            image_size=args.image_size,
            transform=transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=0.5),
        )
        valid = BrainSegmentationDataset(
            images_dir=args.val_data,
            subset="validation",
            image_size=args.image_size,
            random_sampling=False,
        )
    return train, valid


def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
    dsc_list = []
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    for p in range(len(num_slices)):
        y_pred = np.array(validation_pred[index: index + num_slices[p]])
        y_true = np.array(validation_true[index: index + num_slices[p]])
        dsc_list.append(dsc(y_pred, y_true))
        index += num_slices[p]
    return dsc_list


def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)


def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)


def snapshotargs(args):
    args_file = os.path.join(args.logs, "args.json")
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for segmentation of brain MRI"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--vis-images",
        type=int,
        default=200,
        help="number of visualization images to save in log file (default: 200)",
    )
    parser.add_argument(
        "--vis-freq",
        type=int,
        default=10,
        help="frequency of saving images to log file (default: 10)",
    )
    parser.add_argument(
        "--weights", type=str, default="./weights", help="folder to save weights"
    )
    parser.add_argument(
        "--logs", type=str, default="./logs", help="folder to save logs"
    )
    parser.add_argument(
        "--images", type=str, default="./kaggle_3m", help="root folder with images"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="target input image size (default: 256)",
    )
    parser.add_argument(
        "--aug-scale",
        type=int,
        default=0.05,
        help="scale factor range for augmentation (default: 0.05)",
    )
    parser.add_argument(
        "--aug-angle",
        type=int,
        default=15,
        help="rotation angle range in degrees for augmentation (default: 15)",
    )
    parser.add_argument(
        "--train_data", type=str, default="./data/brain-segmentation/train", help="root folder with train data"
    )
    parser.add_argument(
        "--val_data", type=str, default="./data/brain-segmentation/val", help="root folder with val data"
    )
    parser.add_argument(
        "--test_data", type=str, default="./data/brain-segmentation/test", help="root folder with test data"
    )
    parser.add_argument(
        '--use_distance_map', type=bool, default=False, help='set to true to use distance map rather than dice loss'
    )
    parser.add_argument(
        '--use_bce_loss', type=bool, default=False, help='set to true to use binary cross entropy loss'
    )
    parser.add_argument(
        '--bce_pos_weight', type=float, default=1, help='weight applied to positive pixels'
    )
    parser.add_argument(
        '--distance_map_exp', type=float, default=1, help='exponent to use for distances in distance map loss'
    )
    parser.add_argument(
        '--load_weights', type=str, default=None, help='path to weights to lode before training'
    )
    args = parser.parse_args()
    main(args)
