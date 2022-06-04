import argparse
from pathlib import Path

from logzero import logger
import pytorch_lightning as pl

from dentaku_bart import DentakuBart
from dentaku_bart_meta_learning import DentakuBartMetaLearning
from dentaku_bart_maml import DentakuBartMAML

DIR_NAME2MODEL_NAME = {
    "maml": "DentakuBartMAML",
    "maml_fine_tuning": "DentakuBart",
    "meta_learning": "DentakuBartMetaLearning",
    "no_meta_learning": "DentakuBart",
    "pretrain": "DentakuBart",
}


def path2model_name(ckpt_file_path):
    learning_type_dir_name = ckpt_file_path.parts[-2]
    return DIR_NAME2MODEL_NAME.get(learning_type_dir_name)


def get_best_ckpt_path(ckpt_file_path):
    candidate_lsit = []
    for file_path in ckpt_file_path.glob("checkpoint-*.ckpt"):
        splited_path = str(file_path).split("-")
        valid_loss = float(splited_path[3].split("=")[-1])
        candidate_lsit.append(
            {
                "valid_loss": valid_loss,
                "path": file_path  
             }
        )

    return sorted(candidate_lsit, key=lambda d: d["valid_loss"])[0]["path"]



def main(args):
    
    if args.model_name is None:
        args.model_name = path2model_name(args.ckpt_file_path)
        assert args.model_name is not None, "Select collect path!"

    cls = globals()[args.model_name]
    if args.ckpt_file_path.is_dir():
        best_ckpt_path = get_best_ckpt_path(args.ckpt_file_path)
    else:
        best_ckpt_path = args.ckpt_file_path
    
    logger.info(f"Load checkpoint form {best_ckpt_path}")
    pl_model = cls.load_from_checkpoint(best_ckpt_path)


    if args.model_weight_save_path is None:
        args.model_weight_save_path = args.ckpt_file_path.with_name("weights")

    args.model_weight_save_path.mkdir(exist_ok=True)
    model_weight_save_path = args.model_weight_save_path / args.model_name
    logger.info(f"Save weight to {model_weight_save_path}")
    pl_model.model.save_pretrained(model_weight_save_path)
    

        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_file_path", help="Select ckpt file path", type=Path)
    parser.add_argument("--model_name", help="Select model name", type=str)
    parser.add_argument("--model_weight_save_path", help="Select model_weight_save_path", type=Path)
    args = parser.parse_args()
    main(args)
