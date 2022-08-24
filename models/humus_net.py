"""
HUMUS-Net: Hybrid Unrolled Multi-scale Network for Accelerated MRI Reconstruction

We use parts of the framework for VarNet in https://github.com/facebookresearch/fastMRI/blob/main/fastmri/pl_modules/varnet_module.py
"""
from models.humus_block import HUMUSBlock
from fastmri.models import NormUnet
import fastmri
import torch.nn as nn
import torch
from typing import List, Tuple, Optional
import torch.utils.checkpoint as checkpoint

class HUMUSNet(nn.Module):
    """
    HUMUS-Net model obtained from unrolling gradient descent iterations in 
    kspace with HUMUS-Block regularization.
    """

    def __init__(
        self,
        num_cascades: int = 8,
        sens_chans: int = 16,
        sens_pools: int = 4,
        mask_center: bool = True,
        num_adj_slices: int = 1,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
            num_adj_slices: Odd integer, number of adjacent slices used as input
                1: single slice
                n: return (n - 1) / 2 slices on both sides from the center slice
        """
        super().__init__()
        self.num_adj_slices = num_adj_slices
        self.use_checkpoint = kwargs['use_checkpoint']
        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        im_size = kwargs['img_size']
        self.cascades = nn.ModuleList(
            [
                VarNetBlock(
                    HUMUSBlock(in_chans=2*self.num_adj_slices, **kwargs), self.num_adj_slices, im_size) for _ in range(num_cascades)
            ]
        )
        
    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        center_slice = (self.num_adj_slices - 1) // 2
        
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()

        for i, cascade in enumerate(self.cascades):
            if self.use_checkpoint:
                kspace_pred = checkpoint.checkpoint(cascade, kspace_pred, masked_kspace, mask, sens_maps)
            else:
                kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)
            
        kspace_pred = torch.chunk(kspace_pred, self.num_adj_slices, dim=1)[center_slice]
        out = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
        return out
    
    @staticmethod
    def load_from_checkpoint(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        hparams = checkpoint['hyper_parameters']
        model = HUMUSNet(
                 num_cascades=hparams['num_cascades'],
                 img_size=hparams['img_size'], 
                 patch_size=hparams['patch_size'], 
                 embed_dim=hparams['embed_dim'], 
                 depths=hparams['depths'], 
                 num_heads=hparams['num_heads'],
                 window_size=hparams['window_size'], 
                 mlp_ratio=hparams['mlp_ratio'], 
                 use_checkpoint=hparams['use_checkpoint'] if 'use_checkpoint' in hparams else True, 
                 resi_connection=hparams['resi_connection'],
                 bottleneck_depth=hparams['bottleneck_depth'],
                 bottleneck_heads=hparams['bottleneck_heads'],
                 conv_downsample_first=hparams['conv_downsample_first'],
                 num_adj_slices=hparams['num_adj_slices'] if 'num_adj_slices' in hparams else 1,
                 sens_chans=hparams['sens_chans'],
            )
        state_dict = checkpoint['state_dict']
        state_dict = {'.'.join(k.split('.')[1:]):v for k, v in state_dict.items() if 'model' in k} 
        model.load_state_dict(state_dict)
        return model
        
class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module, num_adj_slices=1, im_size=None):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()
        self.num_adj_slices = num_adj_slices
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))
        self.im_size = im_size
        

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        _, c, _, _, _ = sens_maps.shape
        return fastmri.fft2c(fastmri.complex_mul(x.repeat_interleave(c // self.num_adj_slices, dim=1), sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        b, c, h, w, _ = x.shape
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).view(b, self.num_adj_slices, c // self.num_adj_slices, h, w, 2).sum(
            dim=2, keepdim=False
        )
    @staticmethod
    def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    @staticmethod
    def chan_complex_to_last_dim(x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()
    
    @staticmethod
    def norm(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape # 1, 3 * 2, h, w
        #x = x.view(b, 2, c // 2 * h * w)
        x = x.view(b, c, h * w) # 1, 6, h * w

        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    @staticmethod
    def unnorm(
        x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean
    
    def pad_width(self, x):
        pad_w = self.im_size[1] - x.shape[-1] 
        if pad_w > 0:
            pad_w_left = pad_w // 2
            pad_w_right = pad_w - pad_w_left
        else:
            pad_w_left = pad_w_right = 0 
        return torch.nn.functional.pad(x, (pad_w_left, pad_w_right, 0, 0), 'reflect'), pad_w_left, pad_w_right
    
    def unpad_width(self, x, pad_w_left, pad_w_right):
        if pad_w_left > 0:
            x = x[:, :, :, pad_w_left:]
        if pad_w_right > 0:
            x = x[:, :, :, :-pad_w_right]
        return x
        
    def apply_model(self, x):
        x = self.complex_to_chan_dim(x)
        if self.im_size is not None:
            x, p_left, p_right = self.pad_width(x)
        x, mean, std = self.norm(x)
        x = self.model(x)
        x = self.unnorm(x, mean, std)
        if self.im_size is not None:
            x = self.unpad_width(x, p_left, p_right)
        x = self.chan_complex_to_last_dim(x)
        return x

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:       
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        model_term = self.sens_expand(
            self.apply_model(self.sens_reduce(current_kspace, sens_maps)), sens_maps
        )
        return current_kspace - soft_dc - model_term
    
class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.
    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.mask_center = mask_center
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_low_frequencies is None or num_low_frequencies == 0:
            # get low frequency line locations and mask them out
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2

        return pad, num_low_frequencies_tensor

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(
                mask, num_low_frequencies
            )
            masked_kspace = batched_mask_center(
                masked_kspace, pad, pad + num_low_freqs
            )

        # convert to image space
        images, batches = self.chans_to_batch_dim(fastmri.ifft2c(masked_kspace))

        # estimate sensitivities
        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        )
    
def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.
    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.
    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask


def batched_mask_center(
    x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.
    Can operate with different masks for each batch element.
    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.
    Returns:
        A mask with the center filled.
    """
    if not mask_from.shape == mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes.")
    if not mask_from.ndim == 1:
        raise ValueError("mask_from and mask_to must have 1 dimension.")
    if not mask_from.shape[0] == 1:
        if (not x.shape[0] == mask_from.shape[0]) or (
            not x.shape[0] == mask_to.shape[0]
        ):
            raise ValueError("mask_from and mask_to must have batch_size length.")

    if mask_from.shape[0] == 1:
        mask = mask_center(x, int(mask_from), int(mask_to))
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, :, :, start:end] = x[i, :, :, start:end]

    return mask