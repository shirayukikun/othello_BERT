import argparse
import time
from pathlib import Path
from typing import List, Dict, Union, Optional
from itertools import islice

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

from pickle_dataset import PickleDataset
from dentaku_tokenizer.tokenizer import BartDentakuTokenizer

def main(args):
    tokenizer = BartDentakuTokenizer.from_pretrained("facebook/bart-base")
    dataset = PickleDataset(
        args.file_path
    )
    data_loader = DataLoader(
        dataset,
        batch_size=4,
        pin_memory=args.pin,
        num_workers=args.num_workers,
        collate_fn=dataset.collator(tokenizer)
    )

    
    for batch in islice(data_loader, 2):
        print(batch["labels"])
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", "-n", type=int, default=4)
    #parser.add_argument("--size", "-s", type=int, required=True, default=)
    parser.add_argument("--pin", "-p", action="store_true")
    parser.add_argument("--file_path", "-f", type=Path)
    args = parser.parse_args()
    main(args)
