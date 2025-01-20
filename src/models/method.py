import torch
from src.train.trainer import BaseTrainer

class CE(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch = kwargs['batch']
        self.num_classes = kwargs['num_classes']

    def training_step(self, batch, batch_idx):
        _, targets, preds, _, loss,_ = super().model_step(batch)
        self.train_metrics.update(preds, targets)

        for idx in range(self.num_classes):
            self.train_TP[idx] += torch.logical_and((preds == idx), (targets == idx)).sum().item()
            self.train_GT[idx] += (targets == idx).sum().item()

        self.log("train/loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch, sync_dist=True)

        return loss
