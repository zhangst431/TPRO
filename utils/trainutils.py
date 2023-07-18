import os
import os.path

import torch.distributed as dist

import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets.luad_histoseg import LUADTestDataset, LUADTrainingDataset, LUADWSSSDataset
from datasets.bcss import BCSSTestDataset, BCSSTrainingDataset, BCSSWSSSDataset


def get_wsss_dataset(cfg):
    MEAN, STD = get_mean_std(cfg.dataset.name)

    transform = {
        "train": A.Compose([
            A.Normalize(MEAN, STD),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(),
            ToTensorV2(transpose_mask=True),
        ]),
        "val": A.Compose([
            A.Normalize(MEAN, STD),
            ToTensorV2(transpose_mask=True),
        ])
    }

    if cfg.dataset.name == "luad":
        train_dataset = LUADWSSSDataset(cfg.dataset.train_root, mask_root=cfg.dataset.mask_root,
                                        transform=transform["train"])
        val_dataset = LUADTestDataset(cfg.dataset.val_root, split="valid", transform=transform["val"])
    elif cfg.dataset.name == "bcss":
        train_dataset = BCSSWSSSDataset(cfg.dataset.train_root, mask_name=cfg.dataset.mask_root,
                                        transform=transform["train"])
        val_dataset = BCSSTestDataset(cfg.dataset.val_root, split="valid", transform=transform["val"])

    return train_dataset, val_dataset


def get_cls_dataset(cfg):
    MEAN, STD = get_mean_std(cfg.dataset.name)

    transform = {
        "train": A.Compose([
            A.Normalize(MEAN, STD),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(),
            ToTensorV2(transpose_mask=True),
        ]),
        "val": A.Compose([
            A.Normalize(MEAN, STD),
            ToTensorV2(transpose_mask=True),
        ]),
    }

    if cfg.dataset.name == "luad":
        train_dataset = LUADTrainingDataset(cfg.dataset.train_root, transform=transform["train"])
        val_dataset = LUADTestDataset(cfg.dataset.val_root, split="valid", transform=transform["val"])
    elif cfg.dataset.name == "bcss":
        train_dataset = BCSSTrainingDataset(cfg.dataset.train_root, transform=transform["train"])
        val_dataset = BCSSTestDataset(cfg.dataset.val_root, split="valid", transform=transform["val"])

    return train_dataset, val_dataset


def get_mean_std(dataset):
    if dataset == "luad":
        norm = [[0.69164956, 0.50502165, 0.73429191], [0.15521939, 0.22774934, 0.18846453]]
    elif dataset == "bcss":
        norm = [[0.66791496, 0.47791372, 0.70623304], [0.1736589,  0.22564577, 0.19820057]]

    return norm[0], norm[1]


def all_reduced(x, n_gpus):
    dist.all_reduce(x)
    x /= n_gpus

