import argparse
from pathlib import Path
import pickle
import json
from typing import List, Dict, Union, Optional
from dataset_generator_base import DatasetGeneratorBase
from othello_record_generator.othello_record import OthelloRecord, OthelloState
from othello_record_generator.othello_tokenizer import (
    OthelloReversePretrainTokenizer,
    OthelloMaskedPretrainTokenizer,
    OthelloEvalScoreRegrssionTokenizer,
    OthelloEvalScoreRegrssionWithAttensionMaskTokenizer,
)
from logzero import logger
from tqdm import tqdm
from utils import sliding_window

__all__ = [
    "OthelloReversePretrainDatasetGenerator",
    "OthelloMaskedPretrainDatasetGenerator",
    "OthelloEvalScoreRegrssionDatasetGenerator",
    "OthelloEvalScoreRegrssionWithAttensionMaskDatasetGenerator",
]


class OthelloReversePretrainDatasetGenerator(DatasetGeneratorBase):
    def __init__(
            self,
            record_file_path:Path,
            save_file_path:Path,
            exclude_dataset_paths=None,
            number_of_data=None
    ):
        super().__init__(
            record_file_path=record_file_path,
            save_file_path=save_file_path,
            exclude_dataset_paths=exclude_dataset_paths,
            number_of_data=number_of_data,
        )

        self.tokenizer = OthelloReversePretrainTokenizer()


    def prepare_data(self):
        count = 0
        with self.tokenized_data_path.open(mode="wb") as f_tok:
            for record in tqdm(self.pickle_loader(self.record_file_path)):
                for before_state, after_state in sliding_window(record.states, 2):
                    if before_state.pass_flag:
                        continue

                    for rot_k in range(4):
                        str_state = str(before_state.board) + str(before_state.move)
                        if self.in_myself(str_state) or self.in_exclude_datasets(str_state):
                            continue
                        self.instance_set.add(str_state)
                        
                        pickle.dump(
                            self.tokenizer(before_state, after_state),
                            f_tok
                        )
                        count += 1
                        if (self.number_of_data is not None) and (count == self.number_of_data):
                            break
                        before_state.rotate()
                        after_state.rotate()

                    if (self.number_of_data is not None) and (count == self.number_of_data):
                            break

                if (self.number_of_data is not None) and (count == self.number_of_data):
                        break
        
        if (self.number_of_data is not None) and (count != self.number_of_data):
            logger.warning(f"Specified number of data is {self.number_of_data} but only {count} was generated...")
        
        self.post_preparation()
        





class OthelloMaskedPretrainDatasetGenerator(DatasetGeneratorBase):
    def __init__(
            self,
            record_file_path:Path,
            save_file_path:Path,
            exclude_dataset_paths=None,
            number_of_data=None
    ):
        super().__init__(
            record_file_path=record_file_path,
            save_file_path=save_file_path,
            exclude_dataset_paths=exclude_dataset_paths,
            number_of_data=number_of_data,
        )

        self.tokenizer = OthelloMaskedPretrainTokenizer()


    def prepare_data(self):
        count = 0
        with self.tokenized_data_path.open(mode="wb") as f_tok:
            for record in tqdm(self.pickle_loader(self.record_file_path)):
                for state in record.states:
                    for rot_k in range(4):
                        str_state = str(state.board) + str(state.move)
                        if self.in_myself(str_state) or self.in_exclude_datasets(str_state):
                            continue
                        self.instance_set.add(str_state)
                        
                        pickle.dump(
                            self.tokenizer(state),
                            f_tok
                        )
                        count += 1
                        if (self.number_of_data is not None) and (count == self.number_of_data):
                            break
                        state.rotate()
                        
                    if (self.number_of_data is not None) and (count == self.number_of_data):
                            break

                if (self.number_of_data is not None) and (count == self.number_of_data):
                        break
        
        if (self.number_of_data is not None) and (count != self.number_of_data):
            logger.warning(f"Specified number of data is {self.number_of_data} but only {count} was generated...")
        
        self.post_preparation()
        


class OthelloEvalScoreRegrssionDatasetGenerator(OthelloMaskedPretrainDatasetGenerator):
    def __init__(
            self,
            record_file_path:Path,
            save_file_path:Path,
            exclude_dataset_paths=None,
            number_of_data=None,
    ):
        super().__init__(
            record_file_path=record_file_path,
            save_file_path=save_file_path,
            exclude_dataset_paths=exclude_dataset_paths,
            number_of_data=number_of_data,
        )

        self.tokenizer = OthelloEvalScoreRegrssionTokenizer()
    

class OthelloEvalScoreRegrssionWithAttensionMaskDatasetGenerator(OthelloEvalScoreRegrssionDatasetGenerator):
    def __init__(
            self,
            record_file_path:Path,
            save_file_path:Path,
            exclude_dataset_paths=None,
            number_of_data=None,
    ):
        super().__init__(
            record_file_path=record_file_path,
            save_file_path=save_file_path,
            exclude_dataset_paths=exclude_dataset_paths,
            number_of_data=number_of_data,
        )

        self.tokenizer = OthelloEvalScoreRegrssionWithAttensionMaskTokenizer()

