"""
Pytorch Lightning module to handle fastMRI data. 
Modified from https://github.com/facebookresearch/fastMRI/blob/master/fastmri/pl_modules/data_module.py
"""
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Optional, Union, List

import fastmri
import pytorch_lightning as pl
import torch
from data.fastmri_data import CombinedSliceDataset, SliceDataset

import numpy as np

def worker_init_fn(worker_id):
    """
    Handle random seeding for all mask_func.
    """
    worker_info = torch.utils.data.get_worker_info()
    data: Union[
        SliceDataset, CombinedSliceDataset
    ] = worker_info.dataset  # pylint: disable=no-member

    # Check if we are using DDP
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True

    # for NumPy random seed we need it to be in this range
    base_seed = worker_info.seed  # pylint: disable=no-member

    if isinstance(data, CombinedSliceDataset):
        for i, dataset in enumerate(data.datasets):
            if (
                is_ddp
            ):  # DDP training: unique seed is determined by worker, device, dataset
                seed_i = (
                    base_seed
                    - worker_info.id
                    + torch.distributed.get_rank()
                    * (worker_info.num_workers * len(data.datasets))
                    + worker_info.id * len(data.datasets)
                    + i
                )
            else:
                seed_i = (
                    base_seed
                    - worker_info.id
                    + worker_info.id * len(data.datasets)
                    + i
                )
            if dataset.transform.mask_func is not None:
                dataset.transform.mask_func.rng.seed(seed_i % (2 ** 32 - 1))
    else:
        if is_ddp:  # DDP training: unique seed is determined by worker and device
            seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
        else:
            seed = base_seed
        if data.transform.mask_func is not None:
            data.transform.mask_func.rng.seed(seed % (2 ** 32 - 1))

class FastMriDataModule(pl.LightningDataModule):
    """
    Data module class for fastMRI data sets.

    This class handles configurations for training on fastMRI data. It is set
    up to process configurations independently of training modules.

    Note that subsampling mask and transform configurations are expected to be
    done by the main client training scripts and passed into this data module.

    For training with ddp be sure to set distributed_sampler=True to make sure
    that volumes are dispatched to the same GPU for the validation loop.
    """

    def __init__(
        self,
        data_path: Path,
        challenge: str,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        combine_train_val: bool = False,
        test_split: str = "test",
        test_path: Optional[Path] = None,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        use_dataset_cache_file: bool = False,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
        train_scanners: Optional[List[str]] = None,
        val_scanners: Optional[List[str]] = None,
        combined_scanner_val: Optional[bool] = False,
        num_adj_slices: Optional[int] = 1,
    ):
        """
        Args:
            data_path: Path to root data directory. For example, if knee/path
                is the root directory with subdirectories multicoil_train and
                multicoil_val, you would input knee/path for data_path.
            challenge: Name of challenge from ('multicoil', 'singlecoil').
            train_transform: A transform object for the training split.
            val_transform: A transform object for the validation split.
            test_transform: A transform object for the test split.
            combine_train_val: Whether to combine train and val splits into one
                large train dataset. Use this for leaderboard submission.
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
            use_dataset_cache_file: Whether to cache dataset metadata. This is
                very useful for large datasets like the brain data.
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
        self.challenge = challenge
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.combine_train_val = combine_train_val
        self.test_split = test_split
        self.test_path = test_path
        self.sample_rate = sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.train_scanners = train_scanners
        self.val_scanners = val_scanners
        self.combined_scanner_val = combined_scanner_val
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
        dataset: Union[SliceDataset, CombinedSliceDataset]
        if is_train and self.combine_train_val:
            data_paths = [
                self.data_path / f"{self.challenge}_train",
                self.data_path / f"{self.challenge}_val",
            ]
            data_transforms = [data_transform, data_transform]
            challenges = [self.challenge, self.challenge]
            sample_rates, volume_sample_rates = None, None  # default: no subsampling
            if sample_rate is not None:
                sample_rates = [sample_rate, sample_rate]
            if volume_sample_rate is not None:
                volume_sample_rates = [volume_sample_rate, volume_sample_rate]
            dataset = CombinedSliceDataset(
                roots=data_paths,
                transforms=data_transforms,
                challenges=challenges,
                sample_rates=sample_rates,
                volume_sample_rates=volume_sample_rates,
                use_dataset_cache=self.use_dataset_cache_file,
                scanner_models=self.train_scanners,
                num_adj_slices=self.num_adj_slices,
            )
        else:
            if data_partition in ("test", "challenge") and self.test_path is not None:
                data_path = self.test_path
            elif data_partition == "val" and self.combined_scanner_val:
                data_path = [
                    self.data_path / f"{self.challenge}_train",
                    self.data_path / f"{self.challenge}_val",
            ]
            else:
                data_path = self.data_path / f"{self.challenge}_{data_partition}"
            
            if data_partition == "train":
                scanner_models = self.train_scanners
            elif data_partition == "val":
                scanner_models = self.val_scanners
            else:
                scanner_models = None

            dataset = SliceDataset(
                root=data_path,
                transform=data_transform,
                sample_rate=sample_rate,
                volume_sample_rate=volume_sample_rate,
                challenge=self.challenge,
                use_dataset_cache=self.use_dataset_cache_file,
                scanner_models=scanner_models,
                num_adj_slices=self.num_adj_slices
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

    def prepare_data(self):
        # call dataset for each split one time to make sure the cache is set up on the
        # rank 0 ddp process. if not using cache, don't do this
        if self.use_dataset_cache_file:
            if self.test_path is not None:
                test_path = self.test_path
            else:
                test_path = self.data_path / f"{self.challenge}_test"
            data_paths = [
                self.data_path / f"{self.challenge}_train",
                self.data_path / f"{self.challenge}_val",
                test_path,
            ]
            data_transforms = [
                self.train_transform,
                self.val_transform,
                self.test_transform,
            ]
            for i, (data_path, data_transform) in enumerate(
                zip(data_paths, data_transforms)
            ):
                sample_rate = self.sample_rate if i == 0 else 1.0
                volume_sample_rate = self.volume_sample_rate if i == 0 else None
                _ = SliceDataset(
                    root=data_path,
                    transform=data_transform,
                    sample_rate=sample_rate,
                    volume_sample_rate=volume_sample_rate,
                    challenge=self.challenge,
                    use_dataset_cache=self.use_dataset_cache_file,
                    num_adj_slices=self.num_adj_slices,
                )

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
            "--challenge",
            choices=("singlecoil", "multicoil"),
            default="singlecoil",
            type=str,
            help="Which challenge to preprocess for",
        )
        parser.add_argument(
            "--test_split",
            choices=("test", "challenge"),
            default="test",
            type=str,
            help="Which data split to use as test split",
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
            "--use_dataset_cache_file",
            default=False,
            action='store_true',
            help="Whether to cache dataset metadata in a pkl file",
        )
        parser.add_argument(
            "--combine_train_val",
            default=False,
            type=bool,
            help="Whether to combine train and val splits for training",
        )
        parser.add_argument(
            '--train_scanners', 
            default=None, 
            nargs='+', 
            type=str, 
            help='Scanner models to train on. All volumes are used by default. For knee data choose from ["Aera", "Skyra", "Biograph_mMR", "Prisma_fit"].  or any combination of those. Aera is 1.5T scanner, all other models are 3.0T. Only fastMRI knee data is supported.'
        )
        parser.add_argument(
            '--val_scanners', 
            default=None, 
            nargs='+', 
            type=str, 
            help='Scanner models to validate on. All volumes are used by default. For knee data choose from ["Aera", "Skyra", "Biograph_mMR", "Prisma_fit"].  or any combination of those. Aera is 1.5T scanner, all other models are 3.0T.  Only fastMRI knee data is supported.'
        )
        parser.add_argument(
            '--combined_scanner_val',
            default=False, 
            action='store_true',
            help='If set, the training set will consist of scanners selected by --train-scanner from the train dataset, and the validation datasets  will combine all target scanners selected by --val-scanner from train and val datasets. If False, only slices with  scanner type in --val_scanner from the validation set is used for validation.'
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
