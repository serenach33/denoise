from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from .processor import DataProcess
from .predict_processor import PredictDataProcess
from src.utils.data_util import *
import shutil
import random
import numpy as np
import os
import torch


class MyDataModule(LightningDataModule):
    def __init__(self, args):
        super(MyDataModule, self).__init__()
        self.args = args
        self.tar_dir = os.path.join(self.args.data_dir, self.args.tarfile)
        tarname = args.tarfile.split(".")[0]
        self.target_dir = os.path.join(self.args.data_dir, f"{tarname}/origin/")
        self.batch_size = self.args.batch_size

    def prepare_data(self):
        print("Prepraing tar file...")
        # NOTE: do not make any state assignment here

        if self.args.is_debug == False:
            if os.path.exists(self.target_dir) and os.path.isdir(self.target_dir):
                print(f"{self.target_dir} 디렉토리가 존재합니다.")
            else:
                print(f"{self.target_dir} 디렉토리가 존재하지 않습니다.")
                shutil.rmtree(self.target_dir, ignore_errors=True)
                os.makedirs(self.target_dir, exist_ok=True)
                print("Extracting tar file...")
                extract_tar(self.tar_dir, self.target_dir)

       # all the files extracted in target_dir

        # not using tarfile for prediction
        elif self.trainer.predicting:
            os.makedirs(self.target_dir, exist_ok=True)
            if len(os.listdir(self.target_dir)) == 0:
                for f in os.listdir(self.tar_dir):
                    shutil.move(self.tar_dir + f"/{f}", self.target_dir)

        else:
            shutil.rmtree(self.target_dir, ignore_errors=True)
            os.makedirs(self.target_dir, exist_ok=True)
            extract_tar(self.tar_dir, self.target_dir)

    def setup(self, stage: str):
        if stage == "fit" or stage is None:

            print(f"Setting up training and validation datasets...")
            self.train_data = DataProcess(
                self.args,
                "train",
                self.target_dir,
                unlabeled=False,
                is_debug=self.args.is_debug,
            )
            print(f"Number of samples in train_data: {len(self.train_data)}")
            
            self.val_data = DataProcess(
                self.args,
                "val", 
                unlabeled=False, 
                target_dir=self.target_dir,
            )
            print(f"Number of samples in val_data: {len(self.val_data)}")

        elif stage == "test":
            print("Setting up test dataset...")
            self.test_data = DataProcess(
                self.args, 
                "test", 
                unlabeled=False, 
                target_dir=self.target_dir
            )
            print(f"Number of samples in test_data: {len(self.test_data)}")

        else:
            print("Setting up predict dataset...")
            self.predict_data = PredictDataProcess(
                self.args, 
                "predict", 
                self.target_dir
            )
            print(f"Number of samples in predict_data: {len(self.predict_data)}")

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, 
                          num_workers=8, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, 
                          num_workers=8, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        if self.args.purpose == 'val_reload':
            return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, 
                            num_workers=8, pin_memory=True, drop_last=True)
        else:
            return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, 
                            num_workers=8, pin_memory=True)
    def predict_dataloader(self):
        return DataLoader(self.predict_data, batch_size=self.batch_size, shuffle=False, num_workers=8)