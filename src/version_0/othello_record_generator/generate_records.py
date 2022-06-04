import argparse
from pathlib import Path
from typing import List, Dict, Union, Optional
import pickle
import random

from tqdm import trange
from logzero import logger

from libedax4py import (
    Edax,
    DEFAULT_BOOK_DATA_PATH,
    DEFAULT_EVAL_DATA_PATH,
    HintList,
)

from libedax4py.libedax import BLACK as EdaxBlack
from libedax4py.libedax import WHITE as EdaxWhite

from othello_record import OthelloRecord, OthelloState
from othello_lib.othello_utils import Color, Disk, Position, Score, ScoreBoard, Board, BitBoard


class OthelloRecordsGenerator:
    def __init__(
            self,
            save_file_path:Path,
            max_records:int,
            level: int,
            black_selection_policy:str,
            white_selection_policy:str,
            book_file_path:Path=DEFAULT_BOOK_DATA_PATH,
            eval_file_path:Path=DEFAULT_EVAL_DATA_PATH,
            seed=42,
    ):
        self.save_file_path = save_file_path
        self.max_records = max_records
        self.book_file_path = book_file_path
        self.eval_file_path = eval_file_path
        self.rand = random.Random(seed)
        self.level = level
        
        self.edax = Edax(
            book_file_path=self.book_file_path,
            eval_file_path=self.eval_file_path,
            level=self.level,
        )

        self.selection_policy_name = {}
        self.selection_policy_name[Color.BLACK] = black_selection_policy
        self.selection_policy_name[Color.WHITE] = white_selection_policy
        
        self.selection_policy = {}
        self.selection_policy[Color.BLACK] = getattr(
            self,
            f"{black_selection_policy}_select"
        )
        self.selection_policy[Color.WHITE] =  getattr(
            self,
            f"{white_selection_policy}_select"
        )
        
        
    def start(self):
        HEIGH, WIDTH = 8, 8
        with self.save_file_path.open(mode="wb") as f:
            try:
                for loop in trange(self.max_records):
                    state_list = []
                    end_flag = False
                    self.edax.edax_init()

                    while True:
                        color = Color.BLACK if self.edax.edax_get_current_player() == EdaxBlack else Color.WHITE
                        
                        board = Board.from_edax(self.edax)
                        score_board = ScoreBoard(HEIGH, WIDTH)
                        legal_moves = BitBoard(HEIGH, WIDTH)

                        end_flag=self.edax.edax_is_game_over()
                        if end_flag:
                            state_list.append(
                                OthelloState(
                                    turn=None,
                                    board=board,
                                    legal_move_board=legal_moves,
                                    score_board=score_board,
                                    move=None,
                                    end_flag=end_flag,
                                    pass_flag=None,
                                )
                            )
                            break

                        
                        pass_flag=not self.edax.edax_can_move()
                        if pass_flag:
                            state_list.append(
                                OthelloState(
                                    turn=color,
                                    board=board,
                                    legal_move_board=legal_moves,
                                    score_board=score_board,
                                    move=None,
                                    end_flag=end_flag,
                                    pass_flag=pass_flag,
                                )
                            )

                            next_color = (not color)
                            color_mark = "*" if next_color == Color.BLACK else "w"
                            board_string = board.to_string_board() + color_mark
                            self.edax.edax_setboard(board_string.encode())
                            #self.edax.edax_new()
                            continue
                        
                        hintlist = HintList()
                        self.edax.edax_hint(100, hintlist)


                        for h in hintlist.hint[1 : hintlist.n_hints + 1]:
                            pos = Position(move=h.move)
                            legal_moves[pos.y][pos.x] = True
                            score_board[pos.y][pos.x] = h.score

                        board = Board.from_edax(self.edax)
                        pos = self.select_move(score_board, color)
                        
                        state_list.append(
                            OthelloState(
                                turn=color,
                                board=board,
                                legal_move_board=legal_moves,
                                score_board=score_board,
                                move=pos,
                                end_flag=end_flag,
                                pass_flag=pass_flag,
                            )
                        )

                        self.edax.edax_move(pos.move)
                        
                    white_count = self.edax.edax_get_disc(EdaxWhite)
                    black_count = self.edax.edax_get_disc(EdaxBlack)

                    if white_count == black_count:
                        record = OthelloRecord(
                            winner=None,
                            loser=None,
                            draw=True,
                            states=state_list,
                            black_select_policy=self.selection_policy_name[Color.BLACK],
                            white_select_policy=self.selection_policy_name[Color.WHITE],
                        )

                    else:
                        winner = Color.WHITE if white_count > black_count else Color.BLACK
                        loser = not winner
                        
                        record = OthelloRecord(
                            winner=winner,
                            loser=loser,
                            draw=False,
                            states=state_list,
                            black_select_policy=self.selection_policy_name[Color.BLACK],
                            white_select_policy=self.selection_policy_name[Color.WHITE],
                        )
                    
                    pickle.dump(record, f)
                    
            except KeyboardInterrupt:
                logger.info(f"{loop + 1} data was saved!")


    def select_move(self, score_board, color):
        return self.selection_policy[color](score_board)
    
        
    def random_select(self, score_board):
        return self.rand.choice(
            [s for s in score_board.to_score_list() if s.score is not None]
        ).pos

    def best_select(self, score_board):
        return max(
            filter(lambda s: s.score is not None, score_board.to_score_list()),
            key=lambda s: s.score,
        ).pos

    def slop_select(self, score_board):
        legal_score_list = [s for s in score_board.to_score_list() if s.score is not None]
        assert len(legal_score_list) > 0, "There is no candidate legal moves..."
        return self.rand.choices(
            legal_score_list,
            weights=[s.score for s in legal_score_list],
            k=1,
        )[0].pos
    
    def __dell__(self):
        self.edax.terminate()
        
        
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--save_file_path",
            help="Select save file path",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--number_of_data",
            help="Specify number of max records",
            type=int,
            required=True
        )
        parser.add_argument(
            "--seed",
            help="Specify seed",
            type=int,
            default=42
        )

        parser.add_argument(
            "--black_selection_policy",
            help="Specify black_selection_policy",
            type=str,
            required=True,
            choices=["random", "best", "slop"],
        )
        parser.add_argument(
            "--white_selection_policy",
            help="Specify white_selection_policy",
            type=str,
            required=True,
            choices=["random", "best", "slop"],
        )

        parser.add_argument(
            "--level",
            help="Select level",
            type=int,
            default=17
        )
        parser.add_argument(
            "--book_file_path",
            help="Select book file",
            type=Path,
            default=DEFAULT_BOOK_DATA_PATH
        )
        parser.add_argument(
            "--eval_file_path",
            help="Select eval file",
            type=Path,
            default=DEFAULT_EVAL_DATA_PATH
        )
        return parser





def main(args):
    gen = OthelloRecordsGenerator(
        save_file_path=args.save_file_path,
        max_records=args.number_of_data,
        level=args.level,
        black_selection_policy=args.black_selection_policy,
        white_selection_policy=args.white_selection_policy,
        book_file_path=args.book_file_path,
        eval_file_path=args.eval_file_path,
        seed=args.seed,
    )
    gen.start()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    OthelloRecordsGenerator.add_args(parser)
    args = parser.parse_args()
    main(args)
