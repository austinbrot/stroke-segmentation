import argparse
import json
import os

import torch.optim as optim
import segmentation_models_pytorch as smp

from dataset import BrainSegmentationDataset as Dataset
from logger import Logger
from loss import DiceLoss
from trainer import Trainer, TrainerConfig
from transform import transforms


def main(args):
    makedirs(args)
    snapshotargs(args)

    dsc_loss = smp.utils.losses.DiceLoss()

    train_data, val_data, test_dataset = datasets(args)

    model = smp.Unet(
        encoder_name='efficientnet-b0',
        in_channels=3,
        classes=1
    )

    optimizer = optim.Adam(model.parameters())

    logger = Logger(args.logs)

    config = TrainerConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        ckpt_path=args.weights,
        save_epochs=1,
        log_frequency=10,
    )

    trainer = Trainer(
        model,
        optimizer,
        dsc_loss,
        config,
        train_data,
        val_data,
        test_dataset,
        logger=logger
    )

    trainer.train()
    logger.close()


def datasets(args):
    train = Dataset(
        images_dir=args.train_data,
        image_size=args.image_size,
        transform=transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=0.5),
    )
    valid = Dataset(
        images_dir=args.val_data,
        image_size=args.image_size,
        random_sampling=False,
    )
    test = Dataset(
        images_dir=args.test_data,
        image_size=args.image_size,
        random_sampling=False,
    )
    return train, valid, test


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
        "--train_data", type=str, default="./data/brain-segmentation/train", help="root folder with train data"
    )
    parser.add_argument(
        "--val_data", type=str, default="./data/brain-segmentation/val", help="root folder with val data"
    )
    parser.add_argument(
        "--test_data", type=str, default="./data/brain-segmentation/test", help="root folder with test data"
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
    args = parser.parse_args()
    main(args)
