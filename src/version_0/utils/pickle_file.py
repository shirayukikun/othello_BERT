from pathlib import Path
import pickle

from more_itertools import ilen

__all__ = ["PickleFileLoader", "PickleFileWriter"]

class PickleFileLoader:
    def __init__(self, file_path:Path):
        self.file_path = Path(file_path)
        
    def __iter__(self):
        with self.file_path.open(mode="rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

    def __len__(self):
        return ilen(self.__iter__())


class PickleFileWriter:
    def __init__(self, file_path:Path):
        self.file_path = Path(file_path)
        self.file_obj = None
    
    def __enter__(self):
        self.file_obj = self.file_path.open(mode="wb")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, obj):
        pickle.dump(obj, self.file_obj)
        
    def close(self):
        self.file_obj.close()



