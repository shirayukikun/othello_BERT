from pathlib import Path
import os
import json
import pickle
from itertools import islice
import random
import sys

from logzero import logger
from tqdm import tqdm
from more_itertools import ilen

from othello_record_generator import othello_record
from utils import argmax, shuffle_pickle_file
from utils.pickle_file import PickleFileWriter, PickleFileLoader
sys.modules["othello_record"] = othello_record


__all__ = ["DatasetCombineder", "DatasetSpliter"]

class DatasetCombineder:
    def __init__(self, args):
        self.save_file_path = args.save_file_path
        self.source_data_file_paths = args.source_data_file_paths
        self.shuffle = args.shuffle
        assert all(path.exists() for path in self.source_data_file_paths), \
            "Path that is not exist in specified file paths..."

    def __call__(self):

        with PickleFileWriter(self.save_file_path) as f:
            for path in tqdm(self.source_data_file_paths):
                for record in tqdm(PickleFileLoader(path), leave=False):
                    assert type(record) is othello_record.OthelloRecord, \
                        "Pickle file dose not contain othello record..."
                    f.write(record)

        if self.shuffle:
            shuffle_pickle_file(self.save_file_path)
        
                        

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--save_file_path",
            help="Specify save_file_path",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--source_data_file_paths",
            help="Specify source_data_file_paths",
            nargs='*',
            type=Path,
            required=True
        )
        parser.add_argument(
            "--shuffle",
            help="Specify whether shuffle combined records or not",
            action="store_true"
        )
        return parser



class DatasetSpliter:
    def __init__(self, args):
        self.source_data_file_path = args.source_data_file_path
        self.save_file_paths = args.save_file_paths
        self.ratios = args.ratios
        
        assert self.source_data_file_path.exists(), \
            f"{self.source_data_file_path} dose not exists..."
        assert len(self.save_file_paths) == len(self.ratios), \
            f"Length of save_file_paths and ratios dose not match..."
        
    def __call__(self):
        number_of_record = len(PickleFileLoader(self.source_data_file_path))
        number_of_splited_record = [
            int(number_of_record * (r / sum(self.ratios)))
            for r in self.ratios
        ]
        
        # 不足分を良い感じに埋める
        while sum(number_of_splited_record) != number_of_record:
            diff_from_real = [
                number_of_record * (ratio / sum(self.ratios)) - num
                for ratio, num in zip(self.ratios, number_of_splited_record)
            ]
            number_of_splited_record[argmax(diff_from_real)] += 1
        
        logger.info(f"Split data to {number_of_splited_record}")

        pickle_loader = PickleFileLoader(self.source_data_file_path)
        for path, num in tqdm(
                zip(
                    self.save_file_paths,
                    number_of_splited_record
                ),
                leave=False
        ):
            with PickleFileWriter(path) as f:
                for record in tqdm(islice(pickle_loader, num)):
                    f.write(record)
                        
        
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--source_data_file_path",
            help="Specify source_data_file_path",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--save_file_paths",
            help="Specify save_file_paths",
            nargs='*',
            type=Path,
            required=True
        )
        parser.add_argument(
            "--ratios",
            help="Specify split ratios",
            nargs='*',
            type=float,
            required=True
        )
        return parser

