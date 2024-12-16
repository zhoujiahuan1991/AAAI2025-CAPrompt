import os.path
import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import build_continual_dataloader

import utils
import warnings


warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')
warnings.filterwarnings('ignore', category=FutureWarning)

def get_args():
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    config = parser.parse_known_args()[-1][0]
    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_caprompt':
        from configs.cifar100_caprompt import get_args_parser
        config_parser = subparser.add_parser('cifar100_caprompt', help='Split-CIFAR100 CAPrompt configs')
    elif config == 'imr_caprompt':
        from configs.imr_caprompt import get_args_parser
        config_parser = subparser.add_parser('imr_caprompt', help='Split-ImageNet-R CAPrompt configs')
    elif config == 'cub_caprompt':
        from configs.cub_caprompt import get_args_parser
        config_parser = subparser.add_parser('cub_caprompt', help='Split-CUB CAPrompt configs')
    else:
        raise NotImplementedError

    get_args_parser(config_parser)
    args = parser.parse_args()
    args.config = config
    return args

def main(args):
    utils.init_distributed_mode(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if 'caprompt' in args.config :
        import trainers.caprompt_trainer as caprompt_trainer
        caprompt_trainer.train(args)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    
    args = get_args()
    print(args)
    main(args)
