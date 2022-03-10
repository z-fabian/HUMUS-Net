from argparse import ArgumentParser

import fastmri
import torch
from fastmri.data import transforms
from models.humus_net import HUMUSNet
from fastmri.pl_modules import MriModule


class HUMUSNetModule(MriModule):

    def __init__(
        self,
        num_cascades: int = 8,
        sens_pools: int = 4,
        sens_chans: int = 16,
        lr: float = 0.0001,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        num_adj_slices: int = 1,
        mask_center: bool = False,
        **kwargs,
    ):
        """
        Pytorch Lightning module to train and evaluate HUMUS-Net. 
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_pools: Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            sens_chans: Number of channels for sensitivity map U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
        """
        if 'num_log_images' in kwargs:
            num_log_images = kwargs['num_log_images']
            kwargs.pop('num_log_images', None)
        else:
            num_log_images = 16
            
        super().__init__(num_log_images)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.sens_pools = sens_pools
        self.sens_chans = sens_chans
        self.num_adj_slices = num_adj_slices
        self.mask_center = mask_center
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.model = HUMUSNet(
            num_cascades=self.num_cascades,
            sens_chans=self.sens_chans,
            sens_pools=self.sens_pools,
            num_adj_slices=self.num_adj_slices,
            mask_center=self.mask_center,
            **kwargs,
        )

        self.loss = fastmri.SSIMLoss()

    def forward(self, masked_kspace, mask):
        return self.model(masked_kspace, mask)

    def training_step(self, batch, batch_idx):
        masked_kspace, mask, target, _, _, max_value, _ = batch
        output = self(masked_kspace, mask)

        target, output = transforms.center_crop_to_smallest(target, output)
        loss = self.loss(
            output.unsqueeze(1), target.unsqueeze(1), data_range=max_value
        )

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        masked_kspace, mask, target, fname, slice_num, max_value, _ = batch
        output = self.forward(
            masked_kspace, mask
        )
        target, output = transforms.center_crop_to_smallest(target, output)

        return {
            "batch_idx": batch_idx,
            "fname": fname,
            "slice_num": slice_num,
            "max_value": max_value,
            "output": output,
            "target": target,
            "val_loss": self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=max_value
            ),
        }

    def test_step(self, batch, batch_idx):
        masked_kspace, mask, _, fname, slice_num, _, crop_size = batch
        output = self(masked_kspace, mask)

        # check for FLAIR 203
        if output.shape[-1] < crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])

        output = transforms.center_crop(output, crop_size)

        return {
            "fname": fname,
            "slice": slice_num,
            "output": output.cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # network params
        parser.add_argument(
            "--embed_dim", 
            default=72, 
            type=int, 
            help="Embedding dimension"
        )
        parser.add_argument(
            "--depths",
            nargs="+",
            default=[2, 2, 2],
            type=int,
            help="Number of STLs per RSTB. The length of this array determines the number of blocks in the downsampling direction. The last block is always bottleneck and does not downsample.",
        )
        parser.add_argument(
            "--num_heads",
            nargs="+",
            default=[3, 6, 12],
            type=int,
            help="Number of attention heads in each RSTB.",
        )
        parser.add_argument(
            "--mlp_ratio", 
            default=2., 
            type=float, 
            help="Ratio of mlp hidden dim to embedding dim. Default: 2"
        )
        parser.add_argument(
            "--window_size", 
            default=8, 
            type=int, 
            help="Window size. Default: 8"
        )
        parser.add_argument(
            "--patch_size", 
            default=1, 
            type=int, 
            help="Patch size. Default: 1"
        )
        parser.add_argument(
            "--resi_connection", 
            default='1conv', 
            type=str, 
            help="The convolutional block before residual connection. '1conv'/'3conv'"
        )
        parser.add_argument(
            "--bottleneck_depth", 
            default=2,
            type=int, 
            help="Number of STLs in bottleneck."
        )
        parser.add_argument(
            "--bottleneck_heads", 
            default=24, 
            type=int, 
            help="Number of attention heads in bottleneck."
        )
        parser.add_argument(
            '--conv_downsample_first', 
            default=False,   
            action='store_true',          
            help='If set, downsample image by 2x first via convolutions before passing it to MUST.',
        )
        parser.add_argument(
            '--use_checkpointing', 
            default=False,   
            action='store_true',          
            help='If set, checkpointing is used to save GPU memory.',
        )

        # training params (opt)
        parser.add_argument(
            "--lr", 
            default=0.0001, 
            type=float, 
            help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma", 
            default=0.1, 
            type=float, 
            help="Amount to decrease step size"
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )
        parser.add_argument(
            "--uniform_train_resolution",
            nargs="+",
            default=None,
            type=int,
            help="If given, training slices will be center cropped / reflection padded to this size to make sure inputs are of the same size.",
        )

        # unrolling params
        parser.add_argument(
            "--num_cascades",
            default=8,
            type=int,
            help="Number of VarNet cascades",
        )
        parser.add_argument(
            "--sens_pools",
            default=4,
            type=int,
            help="Number of pooling layers for sense map estimation U-Net in VarNet",
        )
        parser.add_argument(
            "--sens_chans",
            default=16,
            type=int,
            help="Number of channels for sense map estimation U-Net in VarNet",
        )
        parser.add_argument(
            "--no_center_masking",
            default=False,
            action='store_true',
            help="If set, kspace center is not masked when estimating sensitivity maps.",
        )

        return parser