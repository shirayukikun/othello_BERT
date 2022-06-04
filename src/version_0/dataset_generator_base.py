from pathlib import Path
import json
import pickle
import sys
from othello_record_generator import othello_record
sys.modules["othello_record"] = othello_record

class DatasetGeneratorBase:
    def __init__(
            self,
            record_file_path:Path,
            save_file_path:Path,
            exclude_dataset_paths=None,
            number_of_data=None,
    ):
        self.number_of_data = number_of_data
        self.record_file_path = record_file_path
        self.save_file_path = save_file_path
        self.save_file_path.mkdir(exist_ok=True)
        self.exclude_dataset_paths = exclude_dataset_paths
        
        if self.exclude_dataset_paths is not None:
            self.exclude_dataset_sets = [
                pickle.load((path / "set.pkl").open(mode="rb"))
                for path in self.exclude_dataset_paths
            ]
        else:
            self.exclude_dataset_sets = []

        self.instance_set = set()

        self.tokenized_data_path = self.save_file_path / "tokenized_data.pkl"
        self.basic_info_file_path = self.save_file_path / "basic_info.json"
        self.set_file_path = self.save_file_path / "set.pkl"


        
    def prepare_data(self):        
        raise NotImplementedError()


    def post_preparation(self):
        with self.set_file_path.open(mode="wb") as f_set:
            pickle.dump(self.instance_set, f_set)
                    
        with self.basic_info_file_path.open(mode="w") as f:
            json.dump(
                {
                    "number_of_data": len(self.instance_set),
                    "record_path": str(self.record_file_path),
                    "DatasetGeneratorName": self.__class__.__name__,
                    "collator": "DataCollatorNoPad"
                },
                f
            )


    
    def pickle_loader(self, file_path:Path):
        with file_path.open(mode="rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break
    
    
    def in_exclude_datasets(self, instance):
        return any(instance in instance_set for instance_set in self.exclude_dataset_sets)

    def in_myself(self, instance):
        return instance in self.instance_set

    def isdisjoint(self, instance_set):
        return self.instance_set.isdisjoint(instance_set)
