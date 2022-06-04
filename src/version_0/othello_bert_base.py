import itertools

import torch
import pytorch_lightning as pl
from logzero import logger

from custom_lr_scheduler import WarmupPolynomialLRScheduler


class OthellouBertBase(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("pl_module_setting")
        parser.add_argument("--lr", type=float, required=True)
        parser.add_argument("--end_lr", type=float, required=True)
        parser.add_argument("--warmup_steps_ratio", type=float, required=True)
        parser.add_argument("--power", type=float, required=True)        
    
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.999)
        parser.add_argument("--eps", type=float, default=1e-08)
        parser.add_argument("--weight_decay", type=float, default=0)

        parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)
        parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
        parser.add_argument("--num_hidden_layers", type=int, default=12)
        parser.add_argument("--num_attention_heads", type=int, default=12)
        parser.add_argument("--hidden_size", type=int, default=768)
        
        parser.add_argument("--model_name_or_path", help="Select model name or path", type=str)
        parser.add_argument("--from_scratch", help="Select whether to use pretrained model", action="store_true")
        return parser
    

    def __init__(self, config):
        super().__init__()
        self.automatic_optimization = True
        self.save_hyperparameters(config)        
        self.test_data_idx = 0
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=self.hparams.eps,
            weight_decay=self.hparams.weight_decay
        )

        assert 0 <= self.hparams.warmup_steps_ratio <= 1, \
            "\"args.num_warmup_steps_ratio\" must be 0 <= ratio <=1"
        
        max_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(max_steps * self.hparams.warmup_steps_ratio)
        decay_steps = max_steps - num_warmup_steps
        
        lr_sheduler = WarmupPolynomialLRScheduler(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            end_lr=self.hparams.end_lr,
            decay_steps=decay_steps,
            power=self.hparams.power
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_sheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        


    def overwrite_model_config(self):
        self.model_config.attention_probs_dropout_prob = self.hparams.attention_probs_dropout_prob
        
        self.model_config.hidden_dropout_prob = self.hparams.hidden_dropout_prob
        
        self.model_config.num_hidden_layers = self.hparams.num_hidden_layers
        self.model_config.num_attention_heads = self.hparams.num_attention_heads
        self.model_config.hidden_size = self.hparams.hidden_size
        
        

    def training_step(self, batch, batch_idx=None):
        output = self.model(**batch, output_hidden_states=False)
        self.log("train_loss", output.loss.item(), on_step=True, on_epoch=True)
        return {"loss": output.loss}


    def validation_step(self, batch, batch_idx=None):
        output = self.model(**batch, output_hidden_states=False)        
        self.log(
            "valid_loss",
            output.loss.item(),
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )    
    
    def test_step(self, batch, batch_idx=None):
        output = self.model(**batch, output_hidden_states=False)
        self.log(
            f"test_loss_{self.test_data_idx}",
            output.loss.item(),
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )
