"""
Pytorch Lightning module to handle Stanford 2D/3D FSE datasets. 
Modified from https://github.com/facebookresearch/fastMRI/blob/master/fastmri/pl_modules/data_module.py
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Optional, Union, List

import fastmri
import pytorch_lightning as pl
import torch
from data.stanford.stanford_data import StanfordSliceDataset

import numpy as np

def worker_init_fn(worker_id):
    """
    Handle random seeding for all mask_func.
    """
    worker_info = torch.utils.data.get_worker_info()
    data: SliceDataset = worker_info.dataset  # pylint: disable=no-member

    # Check if we are using DDP
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True

    # for NumPy random seed we need it to be in this range
    base_seed = worker_info.seed  # pylint: disable=no-member

    if is_ddp:  # DDP training: unique seed is determined by worker and device
        seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
    else:
        seed = base_seed
    if data.transform.mask_func is not None:
        data.transform.mask_func.rng.seed(seed % (2 ** 32 - 1))

class StanfordDataModule(pl.LightningDataModule):
    """
    Data module class for Stanford MRI data sets.

    For training with ddp be sure to set distributed_sampler=True to make sure
    that volumes are dispatched to the same GPU for the validation loop.
    """

    def __init__(
        self,
        data_path: Path,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        train_val_seed: int,
        train_val_split: float,
        test_split: str = "test",
        test_path: Optional[Path] = None,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
        num_adj_slices: Optional[int] = 1,
    ):
        """
        Args:
            data_path: Path to root data directory. For example, if knee/path
                is the root directory with subdirectories multicoil_train and
                multicoil_val, you would input knee/path for data_path.
            train_transform: A transform object for the training split.
            val_transform: A transform object for the validation split.
            test_transform: A transform object for the test split.
            train_val_seed: integer random seed used to generate the train-val split
            train_val_split: float between 0.0 and 1.0 that defines the portion of 
                the dataset used for training. Rest is added to the validation set. 
            test_split: Name of test split from ("test", "challenge").
            test_path: An optional test path. Passing this overwrites data_path
                and test_split.
            sample_rate: Fraction of slices of the training data split to use. Can be
                set to less than 1.0 for rapid prototyping. If not set, it defaults to 1.0. 
                To subsample the dataset either set sample_rate (sample by slice) or 
                volume_sample_rate (sample by volume), but not both.
            volume_sample_rate: Fraction of volumes of the training data split to use. Can be
                set to less than 1.0 for rapid prototyping. If not set, it defaults to 1.0. 
                To subsample the dataset either set sample_rate (sample by slice) or 
                volume_sample_rate (sample by volume), but not both.
            batch_size: Batch size.
            num_workers: Number of workers for PyTorch dataloader.
            distributed_sampler: Whether to use a distributed sampler. This
                should be set to True if training with ddp.
            num_adj_slices: Optional; Odd integer, number of adjacent slices to generate as follows
                1: single slice
                n: return (n - 1) / 2 slices on both sides from the center slice
        """
        super().__init__()

        self.data_path = data_path
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.test_split = test_split
        self.test_path = test_path
        self.sample_rate = sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.train_val_seed = train_val_seed
        self.train_val_split = train_val_split
        self.num_adj_slices = num_adj_slices

    def _create_data_loader(
        self,
        data_transform: Callable,
        data_partition: str,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
    ) -> torch.utils.data.DataLoader:
        if data_partition == "train":
            is_train = True
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
            volume_sample_rate = self.volume_sample_rate if volume_sample_rate is None else volume_sample_rate
        else:
            is_train = False
            sample_rate = 1.0
            volume_sample_rate = None  # default case, no subsampling
            
        # if desired, combine train and val together for the train split
        dataset: SliceDataset
        
        if data_partition in ("test", "challenge") and self.test_path is not None:
            data_path = self.test_path
        else:
            data_path = self.data_path

        dataset = StanfordSliceDataset(
            root=data_path,
            data_partition=data_partition,
            train_val_split=self.train_val_split,
            train_val_seed=self.train_val_seed,
            transform=data_transform,
            sample_rate=sample_rate,
            volume_sample_rate=volume_sample_rate,
            num_adj_slices=self.num_adj_slices,
        )

        # ensure that entire volumes go to the same GPU in the ddp setting
        sampler = None
        if self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(dataset)
            else:
                sampler = fastmri.data.VolumeSampler(dataset)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
        )

        return dataloader

    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, data_partition="train")

    def val_dataloader(self):
        return self._create_data_loader(
            self.val_transform, data_partition="val", sample_rate=1.0
        )

    def test_dataloader(self):
        return self._create_data_loader(
            self.test_transform,
            data_partition=self.test_split,
            sample_rate=1.0,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # dataset arguments
        parser.add_argument(
            "--data_path",
            default=None,
            type=Path,
            help="Path to fastMRI data root",
        )
        parser.add_argument(
            "--test_path",
            default=None,
            type=Path,
            help="Path to data for test mode. This overwrites data_path and test_split",
        )
        parser.add_argument(
            "--sample_rate",
            default=None,
            type=float,
            help="Fraction of slices in the dataset to use (train split only). If not given all will be used. Cannot set together with volume_sample_rate.",
        )
        parser.add_argument(
            "--volume_sample_rate",
            default=None,
            type=float,
            help="Fraction of volumes of the dataset to use (train split only). If not given all will be used. Cannot set together with sample_rate.",
        )
        parser.add_argument(
            "--train_val_split",
            default=0.8,
            type=float,
            help="Fraction of data used in the training set. The rest will be used for validation. Sampling is performed over volumes of MRI data.",
        )
        parser.add_argument(
            "--train_val_seed",
            default=0,
            type=int,
            help="Random seed used to generate train-val split.",
        )
            
        # data loader arguments
        parser.add_argument(
            "--batch_size", 
            default=1, 
            type=int, 
            help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=float,
            help="Number of workers to use in data loader",
        )
        parser.add_argument(
            "--num_adj_slices", 
            default=1, 
            type=int, 
            help="Number of adjacent slices used as input. The target always corresponds to the center slice."
        )
        return parser
