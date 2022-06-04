import argparse
from pathlib import Path

import othello_dataset_generator

def main(args):
    assert args.save_file_path.parent.exists(), f"No such dir {args.save_file_path.parent}"
    assert args.record_file_path.exists(), f"No such file {args.record_file_path}"
    
    data_generator_cls = getattr(
        othello_dataset_generator,
        f"Othello{args.task}DatasetGenerator"
    )
    
    data_generator = data_generator_cls(
        record_file_path=args.record_file_path,
        save_file_path=args.save_file_path,
        exclude_dataset_paths=args.exclude_dataset_paths,
        number_of_data=args.number_of_data,
    )
    data_generator.prepare_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude_dataset_paths", help="Select exclude_dataset_paths", nargs='*', type=Path, default=[])
    parser.add_argument("--save_file_path", help="Select save file path", type=Path, required=True)
    parser.add_argument("--record_file_path", help="Select record file path", type=Path, required=True)
    parser.add_argument("--number_of_data", help="Select number of data", type=int)
    parser.add_argument(
        "--task",
        help="Select task",
        type=str,
        required=True,
        choices=[
            "MaskedPretrain",
            "ReversePretrain",
            "EvalScoreRegrssion",
            "EvalScoreRegrssionWithAttensionMask",
        ]
    )
    
    args = parser.parse_args()
    
    main(args)
