import argparse

from othello_trainer import OthelloTrainer

def main(args):
    trainer = OthelloTrainer(args, mode=args.mode)
    trainer(train_only=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = OthelloTrainer.add_args(parser)
    parser.add_argument(
        "--mode",
        help="Select mode",
        type=str,
        required=True,
        choices=["train", "resume"]
    )
    args = parser.parse_args()
    main(args)
