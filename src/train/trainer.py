
import os
import logging
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MetricCollection
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import Precision, Recall, F1Score, MulticlassConfusionMatrix, Specificity
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils.data_util import Get_MelSpec, Get_Fbank, Normalization, Standardization

class BaseTrainer(pl.LightningModule):
    def __init__(
            self,
            criterion,
            optimizer,
            lr,
            wd,
            transform_dict,
            num_classes,
            transform_type,
            custom_model,
            use_normalization,
            use_standardization,
            augment_type,
            augment_dict,
            batch,
            mode,
            save_img_path=None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["custom_model"])
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.wd = wd
        self.num_classes = num_classes
        self.use_standardization = use_standardization
        self.use_normalization = use_normalization
        self.transform_type = transform_type
        self.standardizer = None
        self.normalizer = None

        if self.use_normalization:
            self.normalizer = Normalization()
        self.batch = batch
        self.mode = mode
        self.save_img_path = save_img_path

        if transform_type == "mel":
            self.convert = Get_MelSpec(transform_dict)
        if transform_type == "fbank":
            self.convert = Get_Fbank(transform_dict)

        metrics = MetricCollection([
            Accuracy(task='multiclass', num_classes=self.num_classes, average='macro'),
            Precision(task='multiclass', num_classes=self.num_classes, average='macro'),
            Recall(task='multiclass', num_classes=self.num_classes, average='macro'),
            F1Score(task='multiclass', num_classes=self.num_classes, average='macro'),
            Specificity(task='multiclass', num_classes=self.num_classes, average='macro'),
        ])

        if "binary" in self.mode:
            metrics = MetricCollection([
                Accuracy(task='binary', num_classes=self.num_classes),
                Precision(task='binary', num_classes=self.num_classes),
                Recall(task='binary', num_classes=self.num_classes),
                F1Score(task='binary', num_classes=self.num_classes),
                Specificity(task='binary', num_classes=self.num_classes),
            ])

        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')
        self.val_best_acc = 0


        self.custom_model = custom_model
        if type(self.custom_model).__name__ == "ASTModel":
            self.classifier = deepcopy(self.custom_model.get_classifier())
        else:
            self.classifier = nn.Linear(self.custom_model.final_feat_dim, self.num_classes)

    def setup(self, stage):

        if stage == "fit":
            if self.use_standardization:
                self.standardizer = Standardization(
                    train_loader=self.trainer.datamodule.train_dataloader()
                )
        elif stage == "validata":

            if self.use_standardization:
                self.standardizer = Standardization(
                    train_loader=self.trainer.datamodule.val_dataloader()
                )

        elif stage == "test":

            if self.use_standardization:
                self.standardizer = Standardization(
                    train_loader=self.trainer.datamodule.test_dataloader()
                )

    def model_step(self, batch):
        x, y, filename, split = batch

        if self.normalizer and not self.standardizer:
            self.transform = nn.Sequential(self.normalizer, self.convert)

        if self.standardizer and not self.normalizer:
            self.transform = nn.Sequential(self.standardizer, self.convert)

        if self.normalizer and self.standardizer:
            self.transform = nn.Sequential(self.normalizer, self.standardizer, self.convert)

        if not self.normalizer and not self.standardizer:
            self.transform = nn.Sequential(self.convert)

        x = self.transform(x)

        if self.save_img_path:
            for i in range(len(x)):
                label_dir = f"{y[i].item()}"
                os.makedirs(self.save_img_path, exist_ok=True)
                file_path = os.path.join(
                    self.save_img_path,
                    f"transformed_{filename[i]}_{split[i]}_{label_dir}.png",
                )
                self.save_image(x[i], file_path)

        features = self.custom_model(x)
        raw_output = self.classifier(features)
        loss = self.criterion[0].to(self.device)(raw_output, y)
        preds = torch.argmax(raw_output, dim=1)

        return x, y, preds, features, loss, raw_output
    
    def save_image(self, tensor, filename):
        """
        Save a tensor as an image file.
        """
        # Convert tensor to numpy array
        np_img = tensor.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        np_img = (np_img - np_img.min()) / (np_img.max() - np_img.min())

        # Save as image
        plt.imsave(filename, np_img, cmap="viridis")

    def on_train_epoch_start(self):
        self.train_TP, self.train_GT = [0] * self.num_classes, [0] * self.num_classes

    def training_step(self, batch, batch_idx):
        pass

    def on_train_epoch_end(self):
        
        score_sum = 0.0
        count = 0

        output = self.train_metrics.compute()
        for metric_name, value in output.items():
            self.log(f"{metric_name}", value, on_epoch=True, sync_dist=True)

            if "train/BinaryRecall" in metric_name:
                score_sum += value
                count += 1
            if "train/BinarySpecificity" in metric_name:
                score_sum += value
                count += 1
        mean_score = score_sum / count if count > 0 else 0.0
        self.log("train/Score", mean_score, on_epoch=True, sync_dist=True)

        self.train_metrics.reset()

        class_score, class_score_avg = get_class_score(self.train_TP, self.train_GT)
        self.log("train/class_score_avg", class_score_avg, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.batch, sync_dist=True)       
        for i in range(len(class_score)):
            self.log(f'train/class_acc{i}', class_score[i], on_step=False, on_epoch=True, prog_bar=False, batch_size=self.batch, sync_dist=True)

    def on_validation_epoch_start(self):
        self.val_TP, self.val_GT = [0] * self.num_classes, [0] * self.num_classes

    def validation_step(self, batch, batch_idx):
        _, targets, preds, _, loss, _ = self.model_step(batch)
        self.val_metrics.update(preds, targets)        

        for idx in range(self.num_classes):
            self.val_TP[idx] += torch.logical_and((preds == idx), (targets == idx)).sum().item()
            self.val_GT[idx] += (targets == idx).sum().item()
        
        self.log("val/loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch, sync_dist=True)

    def on_validation_epoch_end(self):
        score_sum = 0.0
        count = 0
        output = self.val_metrics.compute()

        for metric_name, value in output.items():
            if "val/BinaryRecall" in metric_name:
                score_sum += value
                count += 1
            if "val/BinarySpecificity" in metric_name:
                score_sum += value
                count += 1      
        mean_score = score_sum / count if count > 0 else 0.0
        self.log("val/Score", mean_score, on_epoch=True, sync_dist=True)
        self.log_dict(output, on_epoch=True, sync_dist=True)
        
        if "binary" in self.mode:
            current_acc = output['val/BinaryAccuracy']
        else:
            current_acc = output['val/MulticlassAccuracy']    
        
        if isinstance(current_acc, torch.Tensor):
            current_acc = current_acc.item()

        if current_acc > self.val_best_acc:
            self.val_best_acc = current_acc
        
        self.log('val/best_acc', float(self.val_best_acc), on_epoch=True, sync_dist=True)
        print("Best val acc:", self.val_best_acc)
        self.val_metrics.reset()
        
        class_score, class_score_avg = get_class_score(self.val_TP, self.val_GT)
        self.log("val/class_score_avg", class_score_avg, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.batch, sync_dist=True)       
        for i in range(len(class_score)):
            self.log(f'val/class_acc{i}', class_score[i], on_step=False, on_epoch=True, prog_bar=False, batch_size=self.batch, sync_dist=True)
            

    def on_test_epoch_start(self):
        self.test_TP, self.test_GT = [0] * self.num_classes, [0] * self.num_classes
        self.actual = []
        self.deep_feature = []
        self.preds = []
        self.outputs = []

    def test_step(self, batch, batch_idx):

        _, targets, preds, features, loss, raw_output = self.model_step(batch)
        self.test_metrics.update(preds, targets)
        self.log("test/loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch, sync_dist=True)

        # for tsne
        self.deep_feature += features.detach().cpu().numpy().tolist()
        self.actual += targets.detach().cpu().numpy().tolist()
        self.preds += preds.detach().cpu().numpy().tolist()

        self.outputs.append(list(raw_output.softmax(dim=-1).detach().cpu().numpy()))
        
        for idx in range(self.num_classes):
            self.test_TP[idx] += torch.logical_and((preds == idx), (targets == idx)).sum().item()
            self.test_GT[idx] += (targets == idx).sum().item()

    def on_test_epoch_end(self):
        sp, se, score = get_score(self.test_TP, self.test_GT)
        self.class_score, self.class_score_avg = get_class_score(self.test_TP, self.test_GT)

        self.log("test/sp", sp, batch_size=self.batch, sync_dist=True)
        self.log("test/se", se, batch_size=self.batch, sync_dist=True)
        self.log("test/score", score, batch_size=self.batch, sync_dist=True)

        for i in range(len(self.class_score)):
            self.log(
                f"test/class_acc{i}",
                self.class_score[i],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        output = self.test_metrics.compute()
        self.log_dict(output, on_epoch=True, sync_dist=True)
        self.test_metrics.reset()


    def on_predict_epoch_start(self):
        self.files = []
        self.outputs = []

    def predict_step(self, batch, batch_idx):
        self.transform = nn.Sequential(self.convert)
        x, split_index, filename = batch
        x = self.transform(x)
        features = self.custom_model(x)
        output = self.classifier(features)

        self.outputs.append(output)
        self.files.append(filename)

        return output

    def on_predict_epoch_end(self):
        flat_list = [filename for tuple_ in self.files for filename in tuple_]
        unique_filenames = list(set(flat_list))
        flat_output = [output for tuple_ in self.outputs for output in tuple_]
        # unique한 파일이름이 있으니, 원래 파일 리스트에서 인덱스를 찾자
        indexes = []
        for filename in unique_filenames:
            for index, value in enumerate(flat_list):
                if value == filename:
                    indexes.append(index)
            a = []
            for i in range(len(indexes)):
                output = flat_output[indexes[i]]
                a.append(output)

            softmax_list = [torch.nn.functional.softmax(tensor, dim=-1) for tensor in a]
            # print(softmax_list)
            mean_result = torch.stack(softmax_list).mean(dim=0)

            formatted_result = [f"{item:.2f}" for item in mean_result.cpu().numpy()]
            formatted_string = "[" + ", ".join(formatted_result) + "]"
            print(filename, formatted_string)
            indexes = []

    def configure_optimizers(self):
        optim_params = (
            list(self.custom_model.parameters())
            + list(self.classifier.parameters())
        )
        optimizer = optim.Adam(optim_params, lr=self.lr, weight_decay=self.wd)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
        # for param_group in optimizer.param_groups:
        #     print(param_group['lr'])
        # return [optimizer]
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }

def get_score(preds, targets):

    sp = preds[0] / (targets[0] + 1e-10) * 100
    se = sum(preds[1:]) / (sum(targets[1:]) + 1e-10) * 100
    score = (sp + se) / 2.0

    return sp, se, score

def get_class_score(preds, targets):
    cls = [0] * len(targets)
    for i in range(len(preds)):
        cls[i] += preds[i] / (targets[i] + 1e-10) * 100

    score = sum(cls) / len(cls)

    return cls, score