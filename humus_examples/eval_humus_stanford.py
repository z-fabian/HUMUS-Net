"""
Evaluate HUMUS-Net models trained on the Stanford FSE datasets.
Make sure to define the following for the validation dataset:
    --checkpoint_file: path to the saved model checkpoint to be evaluated 
    --data_path: path to the dataset
    --gpus: number of GPUs for validation
    --accelerations: undersampling ratio in kspace domain, 8 in all experiments
    --center_fractions: describes number of center lines used in the mask,  0.04 in all experiments
    --mask_type: random is used in all experiments
    --train_val_split: proportion of data used for training, rest is added to validation dataset
    --train_val_seed: random seed to generate the train-val split
    
Code based on https://github.com/facebookresearch/fastMRI/fastmri_examples/varnet/train_varnet_demo.py
"""

import os, sys
import pathlib
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute())   )

import pytorch_lightning as pl
import torch
from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from  pl_modules.humus_module import HUMUSNetModule

from data.data_transforms import HUMUSNetDataTransform
from pl_modules.stanford_data_module import StanfordDataModule

# Imports for logging and other utility
from pytorch_lightning.plugins import DDPPlugin
import yaml
from utils import load_args_from_config


def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # model
    # ------------
    if args.challenge == 'multicoil':
        model = HUMUSNetModule.load_from_checkpoint(args.checkpoint_file)
        hparams = torch.load(args.checkpoint_file)['hyper_parameters']
        num_adj_slices = hparams['num_adj_slices']
        uniform_train_resolution = hparams['img_size']
    else:
        raise ValueError('Single-coil data not supported.')
    model.eval()
    
    # ------------
    # data
    # ------------
    
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    
    # use fixed masks for val transform
    val_transform = HUMUSNetDataTransform(uniform_train_resolution=uniform_train_resolution, mask_func=mask)
    
    # ptl data module - this handles data loaders
    data_module = StanfordDataModule(
        data_path=args.data_path,
        train_transform=None,
        val_transform=val_transform,
        test_transform=None,
        test_split=None,
        test_path=None,
        sample_rate=None,
        volume_sample_rate=1.0,
        batch_size=1,
        num_workers=4,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
        train_val_seed=args.train_val_seed,
        train_val_split=args.train_val_split,
        num_adj_slices=num_adj_slices,
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, plugins=DDPPlugin(find_unused_parameters=False))
        
    # ------------
    # run
    # ------------
    trainer.validate(model, datamodule=data_module)


def build_args():
    parser = ArgumentParser()

    # basic args
    backend = "ddp"
    num_gpus = 2 if backend == "ddp" else 1
    batch_size = 1
    
    # client arguments
    parser.add_argument(
        '--checkpoint_file', 
        type=pathlib.Path,          
        help='Path to the checkpoint to load the model from.',
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="random",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.04],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[8],
        type=int,
        help="Acceleration rates to use for masks",
    )

    # data config
    parser = StanfordDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        challenge="multicoil",
        mask_type="random",  # random masks for knee data
        batch_size=batch_size,  # number of samples per batch
        test_path=None,  # path for test split, overwrites data_path
        train_val_split=0.8,
        accelerations=[8], # default experimental setup: 8x acceleration
        center_fractions=[0.04]
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        accelerator=backend,  # what distributed version to use
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
    )

    args = parser.parse_args()

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()