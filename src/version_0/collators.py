from pathlib import Path
from logzero import logger

import torch

class DataCollatorNoPad:
    def __init__(self):
        pass
    
    def __call__(self, batch_source):
        batch = {k:[] for k in batch_source[0].keys()}
        for instance in batch_source:
            for k, v in instance.items():
                batch[k].append(v)

        return {k: torch.stack(v) for k, v in batch.items()}





class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch_source):
        batch = {k:[] for k in batch_source[0].keys()}
        for instance in batch_source:
            for k, v in instance.items():
                batch[k].append(v)

        with self.tokenizer.as_target_tokenizer():
            label_batch = self.tokenizer.pad(
                {"input_ids": batch["labels"]},
                return_attention_mask=False
            )
            batch["labels"] = label_batch["input_ids"]
        
        return self.tokenizer.pad(batch, return_attention_mask=True, return_tensors="pt")
        
    

class MetaLearningDataCollator(DataCollator):
    
    def __call__(self, batch_source):
        inner_source = []
        outer_source = []
        for instances in batch_source:
            inner_source.append(instances["inner"])
            outer_source.append(instances["outer"])

        return {
            "inner": super().__call__(inner_source),
            "outer": super().__call__(outer_source)
        }

