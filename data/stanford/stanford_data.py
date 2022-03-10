"""
This code is modified from https://github.com/facebookresearch/fastMRI/blob/master/fastmri/data/mri_data.py
to support the Stanford datasets.
"""
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn

import h5py
import numpy as np
import torch

class StanfordSliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        data_partition: str,
        train_val_split: float = 0.8,
        train_val_seed: int = 0,
        transform: Optional[Callable] = None,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        num_adj_slices: Optional[int] = 1,
    ):
        """
        Args:
            root: Path to the dataset.
            data_partition: a string either 'train' or 'val'
            train_val_split: float between 0.0 and 1.0, the portion of the dataset used for training
            train_val_seed: int random seed used to generate the train-val split
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the volumes should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            num_adj_slices: Optional; Odd integer, number of adjacent slices to generate as follows
                1: single slice
                n: return (n - 1) / 2 slices on both sides from the center slice
        """
        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.transform = transform
        assert num_adj_slices % 2 == 1, "Number of adjacent slices must be odd in SliceDataset" 
        self.num_adj_slices = num_adj_slices
        self.recons_key = "reconstruction_rss" # only multi-coil is supported
        self.examples = []

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        files = list(Path(root).iterdir())
        
        # generate train-val split 
        # sample by volume
        if data_partition in ['train', 'val']:
            # shuffle volumes based on train-val split seed
            # save current random state and restore later
            state = random.getstate()
            random.seed(train_val_seed)
            random.shuffle(files)
            # restore state
            # this is important so that workers generate the same subsampled datasets later
            random.setstate(state)
        
            len_train = round(len(files) * train_val_split)
            if data_partition == 'train':
                files = files[:len_train]
            else:
                assert data_partition == 'val'
                files = files[len_train:]
            
        for fname in sorted(files):
            data = h5py.File(fname, 'r')
            kspace = data['kspace']
            num_slices = kspace.shape[0]

            self.examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(list(set([f[0].stem for f in self.examples])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.examples = [
                example for example in self.examples if example[0].stem in sampled_vols
            ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname, dataslice = self.examples[i]

        with h5py.File(fname, "r") as hf:
            if self.num_adj_slices == 1:
                kspace = hf["kspace"][dataslice]
            else:
                num_slices = hf["kspace"].shape[0]
                slice_idx_list = self._get_slice_indices(dataslice, num_slices)  
                kspace = []
                for idx in slice_idx_list:
                    kspace.append(hf["kspace"][idx])
                kspace = np.concatenate(kspace, axis=0)

            mask = np.asarray(hf["mask"]) if "mask" in hf else None

            target = hf[self.recons_key][dataslice] if self.recons_key in hf else None

            attrs = dict(hf.attrs)
            attrs["padding_left"] = 0
            attrs["padding_right"] = kspace.shape[-1]
            attrs["recon_size"] = target.shape

        if self.transform is None:
            sample = (kspace, mask, target, attrs, fname.name, dataslice)
        else:
            sample = self.transform(kspace, mask, target, attrs, fname.name, dataslice)

        return sample
    
    def _get_slice_indices(self, dataslice, num_slices_in_volume):
        num_slices_per_side = (self.num_adj_slices - 1) // 2
        slice_idx_l, slice_idx_h = dataslice - num_slices_per_side, dataslice + num_slices_per_side
        if slice_idx_l < 0:
            diff = -slice_idx_l
            slice_idx_list = list(range(0, slice_idx_h + 1))
            for _ in range(diff):
                slice_idx_list = [0] + slice_idx_list
        elif slice_idx_h >= num_slices_in_volume:
            diff = num_slices_in_volume - slice_idx_h + 1
            slice_idx_list = list(range(slice_idx_l, num_slices_in_volume))
            for _ in range(diff):
                slice_idx_list = slice_idx_list + [num_slices_in_volume - 1]
        else:
            slice_idx_list = list(range(slice_idx_l, slice_idx_h + 1))
        return slice_idx_list
