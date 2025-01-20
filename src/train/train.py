import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from src.dataloader.loader import MyDataModule
from src.utils.model_util import get_model, get_method

def train(args, logger):
    dataset = MyDataModule(args)
    model = get_model(args)
    method = get_method(args, dataset.target_dir, model)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # if "bianry" in args.mode:
    early_stopping = EarlyStopping(monitor='val/loss', patience=15, mode='min')
    ckpt_loss_callback = ModelCheckpoint(dirpath=args.save_dir, 
                                monitor="val/loss", 
                                mode='min', 
                                filename='minLoss_{epoch:.2f}', 
                                save_top_k=1, 
                                save_last=False,
                                )
    ckpt_score_callback = ModelCheckpoint(dirpath=args.save_dir, 
                                monitor="val/Score", 
                                mode='max', 
                                filename='maxScore_{epoch:.2f}', 
                                save_top_k=1, 
                                save_last=True,
                                )

    callbacks = [lr_monitor, ckpt_loss_callback, ckpt_score_callback, early_stopping]
    # else:
    #     early_stopping = EarlyStopping(monitor='val_MulticlassAccuracy', patience=10, mode='max')
    #     ckpt_callback = ModelCheckpoint(dirpath=args.save_dir, 
    #                                 monitor="val_MulticlassAccuracy", 
    #                                 mode='max', 
    #                                 filename='best_{epoch:.2f}', 
    #                                 save_top_k=1, 
    #                                 save_last=True,
    #                                 )
        
    if not args.test and not args.predict:
        trainer = Trainer(max_epochs=args.epoch, 
                        num_sanity_val_steps=0, 
                        log_every_n_steps=10, 
                        callbacks=callbacks,
                        logger=logger,
                        devices=args.gpus,
                        strategy=DDPStrategy(find_unused_parameters=True)
                        )     
    
    trainer.fit(method, datamodule=dataset)

    if args.test:
        
        trainer = Trainer()
        
        if args.transform_type == 'mel':
            state_dict = torch.load(args.ckpt_path)
            state_dict = state_dict['state_dict']
            for key in list(state_dict.keys()):
                if 'transform' in key:
                    del state_dict[key]
            method.load_state_dict(state_dict)
            trainer.test(model=method,
                        datamodule=dataset,
                        ckpt_path=None)
            
        else:
            trainer.test(model=method,
                        datamodule=dataset,
                        ckpt_path=args.ckpt_path)