import argparse
from pathlib import Path
from typing import List, Dict, Union, Optional
import os
import json
import math
from datetime import datetime

from logzero import logger
import pytorch_lightning as pl
#from pytorch_lightning.loggers.wandb import WandbLogger

from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

from dentaku_trainer import DentakuTrainer


class DentakuTuner(DentakuTrainer):
    @staticmethod
    def add_args(parent_parser):
        initial_parser = argparse.ArgumentParser()
        initial_parser.add_argument("--tune_method", type=str, required=True, choices=["PBT", "ASH", "Optuna"])
        initial_args, _ = initial_parser.parse_known_args()

        parent_parser = DentakuTrainer.add_args(parent_parser)
        parser = parent_parser.add_argument_group("Tune_configs")

        parser.add_argument("--tune_method", type=str, required=True, choices=["PBT", "ASH", "Optuna"])
        parser.add_argument("--tune_space_file_path", type=Path, required=True)
        parser.add_argument("--cpus_per_trial", type=float, required=True)
        parser.add_argument("--gpus_per_trial", type=float, required=True)
        parser.add_argument("--tune_num_samples", type=int, required=True)

        
        if initial_args.tune_method == "ASH":
            parser.add_argument("--tune_num_max_report", type=int, required=True)
            parser.add_argument("--grace_period", type=int, required=True)
            parser.add_argument("--reduction_factor", type=float, default=4)
            
        elif initial_args.tune_method == "PBT":
            parser.add_argument("--perturbation_interval", type=int, required=True)
            parser.add_argument("--quantile_fraction", type=float, default=0.25)
            parser.add_argument("--resample_probability", type=float, default=0.25)
        elif initial_args.tune_method == "Optuna":
            pass
        else:
            raise NotImplementedError()
        
        return parent_parser
    
    def __init__(self, args, checkpoint_dir=None):
        
        self.tune_ckpt_filename = "tune_checkpoint"
        if checkpoint_dir is not None:
            self.checkpoint_path = Path(checkpoint_dir) / self.tune_ckpt_filename
        else:
            self.checkpoint_path = None

        super().__init__(args, mode="train")
        

        
    def pl_trainer_setting(self):
        start_time = datetime.today().strftime("%Y%m%d%H%M%S")
        
        if self.args.tune_method in  ["ASH", "Optuna"]:
            callbacks = [
                TuneReportCallback(
                    {
                        "valid_loss": "valid_loss",
                        "valid_accuracy": "valid_accuracy", 
                    },
                    on="validation_end"
                ),
            ]
            
        elif self.args.tune_method in  ["PBT"]:
            os.environ["RAY_ALLOW_SLOW_STORAGE"] = "1"
            
            callbacks = [
                TuneReportCheckpointCallback(
                    {
                        "valid_loss": "valid_loss",
                        "valid_accuracy": "valid_accuracy", 
                    },
                    filename=self.tune_ckpt_filename,
                    on="validation_end"
                )
            ]
        else:
            raise NotImplementedError()
            
        
        if self.args.max_epochs == -1:
            self.args.max_epochs = None

        
        self.pl_trainer = pl.Trainer(
            check_val_every_n_epoch=self.args.check_val_every_n_epoch,
            val_check_interval=self.args.val_check_interval,
            deterministic=True,
            callbacks=callbacks,
            gpus=math.ceil(self.args.gpus_per_trial),
            accelerator="gpu",
            fast_dev_run=self.args.fast_dev_run,
            log_every_n_steps=50,
            max_epochs=self.args.max_epochs,
            max_steps=None,
            profiler=None,
            reload_dataloaders_every_n_epochs=0,
            weights_save_path=self.args.weights_save_path,
            default_root_dir=self.args.default_root_dir,
            #logger=pl_logger,
            progress_bar_refresh_rate=0,
        )


    def train(self):
        self.pl_trainer.fit(
            self.pl_model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.valid_dataloader,
            ckpt_path=self.checkpoint_path,
        )
        
        
    def test(self):
        raise NotImplementedError()
    
    def __call__(self):
        self.train()
        
    

