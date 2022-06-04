from pathlib import Path
from typing import List, Dict, Union, Optional
from dataclasses import dataclass

from othello_lib.othello_utils import Color, Disk, Position, Score, ScoreBoard, Board, BitBoard


@dataclass
class OthelloState:
    turn: bool
    board: Board
    legal_move_board: BitBoard
    score_board: ScoreBoard
    move: Position
    end_flag: bool
    pass_flag: bool

    def rotate(self, k=1):
        self.board.rotate(k)
        self.legal_move_board.rotate(k)
        self.score_board.rotate(k)
        height, width = self.board.size()
        if self.move is not None:
            self.move.rotate(height, width, k)
        

@dataclass
class OthelloRecord:
    winner: bool
    loser: bool
    draw: bool
    states: List[OthelloState]
    black_select_policy: str
    white_select_policy: str

    def __getitem__(self, item):
        return self.states[item]


    def rotate(self, k=1):
        for state in self.states:
            state.rotate(k)
