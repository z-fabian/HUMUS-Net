"""
Utilities for example scripts.
"""
import pathlib
import yaml

def load_args_from_config(args):
    config_file = args.config_file
    if config_file.exists():
        with config_file.open('r') as f:
            d = yaml.safe_load(f)
            for k,v in d.items():
                setattr(args, k, v)
    else:
        print('Config file does not exist.')
    return args