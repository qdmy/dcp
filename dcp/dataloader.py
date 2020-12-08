import os

import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import dcp.transforms as group_transforms
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from datetime import datetime
from dcp.dataset import BusDeriverDataset, SegmentSampler, BusDeriverDataset3D, is_high_quality

from dcp.utils.sampler import *


def get_cifar_dataloader(dataset, batch_size, n_threads=4, data_path='/home/dataset/', logger=None):
    """
    Get dataloader for cifar10/cifar100
    :param dataset: the name of the dataset
    :param batch_size: how many samples per batch to load
    :param n_threads:  how many subprocesses to use for data loading.
    :param data_path: the path of dataset
    :param logger: logger for logging
    """

    logger.info("|===>Get datalaoder for " + dataset)

    if dataset == 'cifar10':
        norm_mean = [0.49139968, 0.48215827, 0.44653124]
        norm_std = [0.24703233, 0.24348505, 0.26158768]
    elif dataset == 'cifar100':
        norm_mean = [0.50705882, 0.48666667, 0.44078431]
        norm_std = [0.26745098, 0.25568627, 0.27607843]
    data_root = os.path.join(data_path, 'cifar')

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])

    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=data_root,
                                         train=True,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=data_root,
                                       train=False,
                                       transform=val_transform)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=data_root,
                                          train=True,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=data_root,
                                        train=False,
                                        transform=val_transform)
    else:
        logger.info("invalid data set")
        assert False, "invalid data set"

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=n_threads)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=n_threads)
    return train_loader, val_loader


def get_imagenet_dataloader(dataset, batch_size, n_threads=4, data_path='/home/dataset/', logger=None):
    """
    Get dataloader for imagenet
    :param dataset: the name of the dataset
    :param batch_size: how many samples per batch to load
    :param n_threads:  how many subprocesses to use for data loading.
    :param data_path: the path of dataset
    :param logger: logger for logging
    """

    logger.info("|===>Get datalaoder for " + dataset)

    dataset_path = os.path.join(data_path, dataset)
    traindir = os.path.join(dataset_path, "train")
    valdir = os.path.join(dataset_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_threads,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_threads,
        pin_memory=True)
    return train_loader, val_loader


def get_sub_imagenet_dataloader(dataset, batch_size, num_samples_per_category,
                                n_threads=4, data_path='/home/dataset/', logger=None):
    """
    Get dataloader for imagenet
    :param dataset: the name of the dataset
    :param batch_size: how many samples per batch to load
    :param n_threads:  how many subprocesses to use for data loading.
    :param data_path: the path of dataset
    :param logger: logger for logging
    """

    logger.info("|===>Get datalaoder for " + dataset)

    dataset_path = os.path.join(data_path, "imagenet")
    traindir = os.path.join(dataset_path, "train")
    valdir = os.path.join(dataset_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    stratified_categories_index = get_stratified_categories_index(train_dataset)
    stratified_sampler = StratifiedSampler(stratified_categories_index, num_samples_per_category)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=stratified_sampler,
        num_workers=n_threads,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_threads,
        pin_memory=True)
    return train_loader, val_loader


# add for pcldriver
def get_pcldriver_dataloader(dataset, batch_size, n_threads=16, data_path='/gdata/driver_behaviour/chenzhen/',
                             logger=None, multi_frame=False):
    annotation = 'dcp/data/annotation.json'
    if multi_frame:
        T = group_transforms
        Dataset = BusDeriverDataset3D
    else:
        T = transforms
        Dataset = BusDeriverDataset

    filters = [
        # only train with quality==0, the frames with other quality will disturb the training
        is_high_quality,
    ]

    normalization = T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

    train_transforms = T.Compose([
        T.RandomResizedCrop(size=224, scale=[0.08, 1.0]),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalization,
    ])

    trainset = Dataset(
        root=data_path,
        anno_path=annotation,
        train=True,
        filters=filters,
        transforms=train_transforms,
        n_frames=1,
        interval=0,
    )

    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalization,
    ])
    valset = Dataset(
        root=data_path,
        anno_path=annotation,
        train=False,
        filters=filters,
        transforms=val_transform,
        n_frames=1,
        interval=0
    )

    if multi_frame:
        train_sampler = RandomSampler(trainset)
        val_sampler = SequentialSampler(valset)
    else:
        train_sampler = SegmentSampler(trainset, train=True,
                                       batch_size=batch_size)
        val_sampler = SegmentSampler(valset, train=False, batch_size=batch_size)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=n_threads,
        pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        valset,
        sampler=val_sampler,
        batch_size=batch_size,
        num_workers=n_threads,
        pin_memory=False,
    )
    return train_loader, val_loader