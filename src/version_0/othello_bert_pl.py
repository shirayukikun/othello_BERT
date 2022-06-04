from logzero import logger
import torch
import pytorch_lightning as pl
from transformers import BertForTokenRegression, BertForMaskedLM, BertConfig

from othello_bert_base import OthellouBertBase

__all__ = ["OthelloBertMaskedLM", "OthelloBertTokenRegression"]

class OthelloBertMaskedLM(OthellouBertBase):
    @staticmethod
    def add_model_specific_args(parent_parser):
        OthellouBertBase.add_model_specific_args(parent_parser)
        return parent_parser
    
    def __init__(self, config):
        super().__init__(config)

        if self.hparams.from_scratch:
            self.model_config = BertConfig.from_pretrained(self.hparams.model_name_or_path)
            self.overwrite_model_config()
            self.model = BertForMaskedLM(config=self.model_config)
            logger.info("Learning from scrach!")
            
        else:
            self.model_config = BertConfig.from_pretrained(self.hparams.model_name_or_path)
            self.overwrite_model_config()
            self.model = BertForMaskedLM.from_pretrained(
                self.hparams.model_name_or_path,
                config=self.model_config
            )
            logger.info(f"Load pretrained model from \"{self.hparams.model_name_or_path}\"")



class OthelloBertTokenRegression(OthellouBertBase):
    @staticmethod
    def add_model_specific_args(parent_parser):
        OthellouBertBase.add_model_specific_args(parent_parser)
        return parent_parser
    
    def __init__(self, config):
        super().__init__(config)

        if self.hparams.from_scratch:
            self.model_config = BertConfig.from_pretrained(self.hparams.model_name_or_path)
            self.overwrite_model_config()
            self.model = BertForTokenRegression(config=self.model_config)
            logger.info("Learning from scrach!")
            
        else:
            self.model_config = BertConfig.from_pretrained(self.hparams.model_name_or_path)
            self.overwrite_model_config()
            self.model = BertForTokenRegression.from_pretrained(
                self.hparams.model_name_or_path,
                config=self.model_config
            )
            logger.info(f"Load pretrained model from \"{self.hparams.model_name_or_path}\"")
