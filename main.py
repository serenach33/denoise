import os
import json
import time
import random
import torch
import numpy as np
import pytorch_lightning as pl
from src.utils import args
from src.utils.logger import setup_wandb_logger
from src.train import train

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    if args.no_logging:
        logger = None
    else:
        logger = setup_wandb_logger(args)

    if args.test or args.predict:
        train.train(args, None)
    else:
        train.train(args, logger)

if __name__ == "__main__":
    args = args.parser.parse_args()
    args.local_time = f"{time.localtime().tm_year:04d}{time.localtime().tm_mon:02d}{time.localtime().tm_mday:02d}{time.localtime().tm_hour:02d}{time.localtime().tm_min:02d}{time.localtime().tm_sec:02d}"
    args.save_dir = (
        f"{args.save_dir}/{args.mode}/{args.backbone}_{args.method}/{args.local_time}"
    )
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    AVAIL_GPUS = torch.cuda.device_count()
    GPU_TYPE = torch.cuda.get_device_name()
    NUM_WORKERS = int(os.cpu_count() / AVAIL_GPUS)

    print("*" * 20)
    print(f"\nTraining with {AVAIL_GPUS} {GPU_TYPE} GPU(s)\n")
    print(f"Arguments and Checkpoints are saved in {args.save_dir}\n")
    print("*" * 20)

    args.gpus = AVAIL_GPUS
    args.workers = NUM_WORKERS

    if args.gpus > 1:
        args.sync_dist = True
    else:
        args.sync_dist = False

    with open(os.path.join(args.save_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    main(args)