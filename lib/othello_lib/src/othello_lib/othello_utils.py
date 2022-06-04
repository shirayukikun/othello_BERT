from pathlib import Path
from typing import List, Dict, Union, Optional
from dataclasses import dataclass
from itertools import chain
from collections import namedtuple
from typing import Union

import numpy as np

from libedax4py import libedax, edaxutil
from libedax4py import Board as EdaxBoard
from libedax4py import Position as EdaxPosition

class Color:
    BLACK = True
    WHITE = False


class Disk:
    BLACK = Color.BLACK
    WHITE = Color.WHITE



class Position:
    
    def __init__(self, x=None, y=None, move=None):
        assert (x is None and y is None) or (move is None), \
            "Specify x, y coordinate or move"

        if move is None:
            self.x = x
            self.y = y
            self.move = self.to_move()

        elif x is None and y is None:
            if type(move) is str:
                self.move = move
            else:
                self.move = edaxutil.moveToString(move)
                
            self.x, self.y = self.to_coordinate()
            
        else:
            raise NotImplementedError()
        

    def to_move(self):
        return (chr(self.x + ord("a")) + str(self.y + 1)).encode()

    
    def to_coordinate(self):
        x = ord(self.move[0]) - ord("a")
        y = int(self.move[1]) - 1    
        return x, y

    def __str__(self):
        return f"{self.__class__.__name__}({self.x}, {self.y})"


    def rotate(self, height, width, k):
        for _ in range(k % 4):
            temp_y = self.y
            temp_x = self.x
            self.x = temp_y
            self.y = width - 1 - temp_x
            
        self.move = self.to_move()
        

        
    

@dataclass
class Score:
    pos: Position
    score: Union[int, float]
    


class BoardBase:
    def __init__(self, dtype, height, width):
        self.__board = [
            [dtype() for i in range(width)] for j in range(height)
        ]
        self.height = height
        self.width = width
        
    def size(self):
        return self.height, self.width

    def __getitem__(self, item):
        return self.__board[item]

    def to_list(self):
        return self.__board

    def __len__(self):
        return len(self.__board)

    def __iter__(self):
        return iter(self.__board)


    def show(self):
        print("\n".join(str(row) for row in self.__board))
    
    def rotate(self, k=1):
        self.__board = list(np.rot90(self.__board, k))
        if k % 2 == 1:
            self.height, self.width = self.width, self.height 
        


class BitBoard(BoardBase):
    def __init__(self, height, width):
        super().__init__(bool, height, width)


class ScoreBoard(BoardBase):
    def __init__(self, height, width):
        super().__init__(lambda: None, height, width)
        
    def to_score_list(self):
        height, width = self.size()
        score_list = []
        for y in range(height):
            for x in range(width):
                score_list.append(
                    Score(
                        pos=Position(x=x, y=y),
                        score=self[y][x]
                    )
                )
        
        return score_list
                        
        
    
class Board:
    def __init__(self, black:List[List[bool]]=None, white:List[List[bool]]=None):
        if black is None:
            self.black = [[False for i in range(8)] for j in range(8)]
        else:
            self.black = black

        if white is None:
            self.white = [[False for i in range(8)] for j in range(8)]
        else:
            self.white = white


            
    def size(self):
        assert len(self.black) == len(self.white), "Not same height..."
        assert len(self.black[0]) == len(self.white[0]), "Not same width..."
        assert all(
            map(
                lambda l: len(self.black[0]) == len(l), self.black
            )
        ), "Not all widths are same..."
        assert all(
            map(
                lambda l: len(self.white[0]) == len(l), self.white
            )
        ), "Not all widths are same..."
        return len(self.black), len(self.black[0])


    def to_string_board(self):
        return "".join(self._to_string_board_generator())

    
    def _to_string_board_generator(self):
        for b, w in zip(
                chain.from_iterable(self.black),
                chain.from_iterable(self.white),
        ):
            yield self.get_piece(b, w)

    def __str__(self):
        return "\n".join(
            "".join(
                self.get_piece(b, w) for b, w in zip(black_line, white_line)
            )
            for black_line, white_line in zip(self.black, self.white)
        )
            

    def get_piece(self, black_flag, white_flag):
        assert not(black_flag and white_flag), "Both black and white are true..."
        if not black_flag and not white_flag:
            return "-"
        elif black_flag:
            return "x"
        elif white_flag:
            return "o"
        else:
            raise RuntimeError()

    def count(self):
        return sum(chain.from_iterable(self.black)), sum(chain.from_iterable(self.white))


    def rotate(self, k=1):
        self.black.rotate(k)
        self.white.rotate(k)


    def rotate(self, k=1):
        self.black = list(np.rot90(self.black, k))
        self.white = list(np.rot90(self.white, k))

        
    @classmethod
    def from_edax(cls, edax_obj):
        board = cls()
        edax_board = EdaxBoard()
        edax_obj.edax_get_board(edax_board)

        currentPlayer = edax_obj.edax_get_current_player()
        
        player_color = Disk.BLACK if (currentPlayer == libedax.BLACK) else Disk.WHITE
        opponent_color = not player_color
        
        for i in range(8):
            for j in range(8):
                if edax_board.player & (1 << (j + 8 * i)):
                    if player_color == Disk.BLACK:
                        board.black[i][j] = True
                    elif player_color == Disk.WHITE:
                        board.white[i][j] = True
                    else:
                        raise RuntimeError()
                elif edax_board.opponent & (1 << (j + 8 * i)):
                    if opponent_color == Disk.BLACK:
                        board.black[i][j] = True
                    elif opponent_color == Disk.WHITE:
                        board.white[i][j] = True
                    else:
                        raise RuntimeError()
                    
        return board


