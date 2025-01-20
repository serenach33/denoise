from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.utilities.rank_zero import rank_zero_only


@rank_zero_only

def setup_wandb_logger(args):
    # Create and return the WandbLogger instance
    wandb_logger = WandbLogger(
        name=f"{args.local_time}_{args.description}_{args.backbone}_{args.method}",
        project=args.project_name,
        group=args.group,
        id=f"{args.local_time}",
    )
    return wandb_logger
