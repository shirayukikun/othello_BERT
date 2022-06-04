import argparse
from pathlib import Path
from itertools import islice
import pickle
import random
import json

from more_itertools import ilen, chunked, ichunked
from logzero import logger
from tqdm import tqdm, trange

from dataset_generator_base import DatasetGeneratorBase


class WithoutCalculationDatasetConverter(DatasetGeneratorBase):
    def __init__(
        self,
        original_data_set_file_path:Path,
        number_of_data:int,
        save_file_path:Path,
    ):
        self.original_data_set_file_path = original_data_set_file_path
        self.number_of_data = number_of_data
        self.save_file_path = save_file_path


    def prepare_data(self):
        self.save_file_path.mkdir(exist_ok=True)

        with (self.original_data_set_file_path / "basic_info.json").open(mode="r") as f:
            original_basic_info = json.load(f)
            
        
        # Save basic info
        self.basic_info_file_path = self.save_file_path / "basic_info.json"
        with self.basic_info_file_path.open(mode="w") as f:
            if self.number_of_data is not None:
                original_basic_info["number_of_data"] = self.number_of_data
            assert original_basic_info["number_of_data"] >= self.number_of_data, \
                "Specified number of data grater than number of original dataset..."
            json.dump(
                original_basic_info,
                f
            )

        self.raw_data_file_path = self.save_file_path / "raw_data.tsv"
        original_raw_data_file_path = self.original_data_set_file_path / "raw_data.tsv"
        
        
        with self.raw_data_file_path.open(mode="w") as f_raw, \
             original_raw_data_file_path.open(mode="r") as f_orig:

            for line in tqdm(islice(f_orig, self.number_of_data)):
                f_raw.write(self.convert_instance(line))
        
        
    def convert_instance(self, line):
        passage, question, answer = line.split("\t")
        #a = 1 + 2, c = 3 + 4\ta=\t3
        var = question[0]
        list_passage = list(passage)
        start = list_passage.index(var)
        try:
            end = list_passage.index(",", start)
        except ValueError:
            end = len(list_passage)
            
        formula = "".join(list_passage[start:end])
        return "\t".join((passage, question, formula[4:] + "\n"))
