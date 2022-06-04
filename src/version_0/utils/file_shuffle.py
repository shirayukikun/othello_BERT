import random
from pathlib import Path
import pickle
import tempfile
from datetime import datetime
import shutil

from tqdm import tqdm
from more_itertools import ilen, chunked

from .pickle_file import PickleFileWriter, PickleFileLoader

def large_file_shuffle(input_file_path:Path, output_file_path:Path, batch_size=100000, seed=42):
    random.seed(seed)
    with input_file_path.open(mode="r") as f:
        line_count = ilen(f)
    
    indexes = list(range(line_count))
    random.shuffle(indexes)

    with output_file_path.open(mode="w") as f_output:

        for index_batch in chunked(indexes, batch_size):
            batch = {}
            index_batch_set = set(index_batch)
            with input_file_path.open(mode="r") as f_input:
                for i, line in enumerate(f_input):
                    if i in index_batch_set:
                        batch[i] = line

                for i in index_batch:
                    f_output.write(batch[i])

    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = Path(temp_dir.name)

                    

def shuffle_pickle_file(
        file_path: Path,
        output_file_path:Path =None,
        batch_size: int =10000,
        seed=42
):
    assert file_path.exists(), f"No such file or directory {file_path}"

    random_module = random.Random(seed)
    data_size = ilen(PickleFileLoader(file_path))
    indexes = list(range(data_size))
    random_module.shuffle(indexes)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        now = datetime.today().strftime("%Y%m%d%H%M%S")
        temp_file_path = Path(temp_dir) / f"shuffle_temp_{now}.pkl"

        with PickleFileWriter(temp_file_path) as f_temp:
            for index_batch in tqdm(chunked(indexes, batch_size)):
                batch = {}
                index_batch_set = set(index_batch)

                for i, data in enumerate(PickleFileLoader(file_path)):
                    if i in index_batch_set:
                        batch[i] = data

                for i in index_batch:
                    f_temp.write(batch[i])

        if output_file_path is None:
            shutil.move(
                temp_file_path,
                file_path,
                copy_function=shutil.copy
            )
        else:
            shutil.move(
                temp_file_path,
                output_file_path,
                copy_function=shutil.copy
            )
