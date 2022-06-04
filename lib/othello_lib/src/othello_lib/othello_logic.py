from pathlib import Path
from typing import List, Dict, Union, Optional

from libedax4py import libedax, edaxutil, HintList, Edax
from libedax4py import Board as EdaxBoard
from libedax4py import Position as EdaxPosition

from .othello_utils import Board, BitBoard, Position, Disk



class OthelloLogic:
    def __init__(self):
        self.edax = Edax(level=0)
        
    def set_edax_board(self, board:Board, color:bool):
        color_mark = "*" if color == Disk.BLACK else "w"
        board_string = board.to_string_board() + color_mark
        self.edax.edax_setboard(board_string.encode())
        self.edax.edax_new()
        

    def put(self, board:Board, pos:Position, color:bool):
        height, width = board.size()
        use_edax = (height == 8 and width == 8)
        
        if use_edax:
            self.set_edax_board(board, color)
            if self.edax.edax_move(pos.move):
                return Board.from_edax(self.edax)
            else:
                return None
        else:
            raise NotImplementedError("Only support 8x8")


    def check_pass(self, board:Board, color:bool):
        height, width = board.size()
        use_edax = (height == width == 8)
        
        if use_edax:
            self.set_edax_board(board, color)
            pass_flag = not self.edax.edax_can_move()
            end_flag =  self.edax.edax_is_game_over()
            
        else:
            raise NotImplementedError("Only support 8x8")

        return pass_flag, end_flag

    def get_hints(self, ai, board:Board, color:bool):
        scores = ai.calc_score(board, color)
        return scores


    def get_leagal_moves(self, board:Board, color:bool):
        height, width = board.size()
        use_edax = (height ==  width == 8)
        leagal_moves = BitBoard(height, width)
        
        if use_edax:
            self.set_edax_board(board, color)
            hintlist = HintList()
            self.edax.edax_hint(100, hintlist)            
            
            for h in hintlist.hint[1 : hintlist.n_hints + 1]:
                pos = Position(move=h.move)
                leagal_moves[pos.y][pos.x] = True
                
        else:
            raise NotImplementedError()
        
        return leagal_moves
    
    def __dell__(self):
        self.edax.terminate()
