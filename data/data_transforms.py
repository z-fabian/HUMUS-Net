"""
Model dependent data transform for HUMUS-Net.
Modified from https://github.com/facebookresearch/fastMRI/blob/master/fastmri/data/transforms.py
"""
from typing import Dict, Optional, Sequence, Tuple, List, Union, NamedTuple
import fastmri
import numpy as np
import torch

from fastmri.data.subsample import MaskFunc
from fastmri.data.transforms import to_tensor, apply_mask, center_crop
from fastmri import fft2c, ifft2c, rss_complex, complex_abs

class HUMUSNetDataTransform:
    """
    Data Transformer for training HUMUS-Net model.
    """

    def __init__(self, 
                 uniform_train_resolution: Union[List[int], Tuple[int]],
                 mask_func: Optional[MaskFunc] = None, 
                 use_seed: bool = True,
):
        """
        Args:
            uniform_train_resolution: Fixed spatial input image resolution.
                Images will be reflection-padded and cropped to this size.
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.uniform_train_resolution = uniform_train_resolution

    def _crop_if_needed(self, image):
        w_from = h_from = 0
        
        if self.uniform_train_resolution[0] < image.shape[-3]:
            w_from = (image.shape[-3] - self.uniform_train_resolution[0]) // 2
            w_to = w_from + self.uniform_train_resolution[0]
        else:
            w_to = image.shape[-3]
        
        if self.uniform_train_resolution[1] < image.shape[-2]:
            h_from = (image.shape[-2] - self.uniform_train_resolution[1]) // 2
            h_to = h_from + self.uniform_train_resolution[1]
        else:
            h_to = image.shape[-2]

        return image[..., w_from:w_to, h_from:h_to, :]
    
    def _pad_if_needed(self, image):
        pad_w = self.uniform_train_resolution[0] - image.shape[-3]
        pad_h = self.uniform_train_resolution[1] - image.shape[-2]
        
        if pad_w > 0:
            pad_w_left = pad_w // 2
            pad_w_right = pad_w - pad_w_left
        else:
            pad_w_left = pad_w_right = 0 
            
        if pad_h > 0:
            pad_h_left = pad_h // 2
            pad_h_right = pad_h - pad_h_left
        else:
            pad_h_left = pad_h_right = 0 
            
        return torch.nn.functional.pad(image.permute(0, 3, 1, 2), (pad_h_left, pad_h_right, pad_w_left, pad_w_right), 'reflect').permute(0, 2, 3, 1)
        
    def _to_uniform_size(self, kspace):
        image = ifft2c(kspace)
        image = self._crop_if_needed(image)
        image = self._pad_if_needed(image)
        kspace = fft2c(image)
        return kspace

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, int, float, torch.Tensor]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.
        Returns:
            tuple containing:
                masked_kspace: k-space after applying sampling mask.
                mask: The applied sampling mask
                target: The target image (if applicable).
                fname: File name.
                slice_num: The slice index.
                max_value: Maximum image value.
                crop_size: The size to crop the final image.
        """
        is_testing = (target is None)
        
        # Make sure data types match
        kspace = kspace.astype(np.complex64)
        
        # Add singleton channel dimension if singlecoil
        if len(kspace.shape) == 2:
            kspace = np.expand_dims(kspace, axis=0)
        # Concatenate slices along channel dim if multi-slice
        elif len(kspace.shape) == 4:
            H, W = kspace.shape[-2:]
            kspace = np.reshape(kspace, (-1, H, W))
        assert len(kspace.shape) == 3
        
        if not is_testing:
            target = target.astype(np.float32)
            target = to_tensor(target)               
            max_value = attrs["max"].astype(np.float32)
        else:
            target = torch.tensor(0)
            max_value = 0.0

        kspace = to_tensor(kspace)
                
        if not is_testing:
            kspace = self._to_uniform_size(kspace)
        else:
            # Only crop image height
            if self.uniform_train_resolution[0] < kspace.shape[-3]:
                image = ifft2c(kspace)
                h_from = (image.shape[-3] - self.uniform_train_resolution[0]) // 2
                h_to = h_from + self.uniform_train_resolution[0]
                image = image[..., h_from:h_to, :, :]
                kspace = fft2c(image)
                
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        if not is_testing:
            crop_size = torch.tensor([target.shape[0], target.shape[1]])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        if self.mask_func:
            masked_kspace, mask = apply_mask(
                kspace, self.mask_func, seed, (acq_start, acq_end)
            )
        else:
            masked_kspace = kspace
            shape = np.array(kspace.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1] * len(shape)
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask = mask.reshape(*mask_shape)
            mask[:, :, :acq_start] = 0
            mask[:, :, acq_end:] = 0
        
        return (            
            masked_kspace,
            mask.byte(),
            target,
            fname,
            slice_num,
            max_value,
            crop_size,
        )