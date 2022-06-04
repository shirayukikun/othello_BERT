from pathlib import Path
import pickle
from itertools import chain

import torch

from othello_lib.othello_utils import Color, Disk, Position, Score, ScoreBoard, Board, BitBoard
from .othello_record import OthelloRecord, OthelloState


class OthelloToken:
    PUT = 6
    LEAGAL = 5
    BLANK = 4
    OPPONENT = 3
    ME = 2
    MASK = 1
    PAD = 0

    

class OthelloTokenizerBase:
    def __init__(self):
        pass

    def __call__(self, *args, **kargs):
        raise NotImplementedError()

    
    def get_token(self, my_flag:bool, opponent_flag:bool):
        assert not(my_flag and opponent_flag), "Both black and white are true..."
        if not my_flag and not opponent_flag:
            return OthelloToken.BLANK
        elif my_flag:
            return OthelloToken.ME
        elif opponent_flag:
            return OthelloToken.OPPONENT
        else:
            raise RuntimeError()





        
class OthelloReversePretrainTokenizer(OthelloTokenizerBase):
    def __init__(self):
        super().__init__()


    def __call__(self, before_state:OthelloState, after_state:OthelloState):
        HEIGH, WIDTH = before_state.board.size()
        my_color = before_state.turn
        
        if my_color == Color.BLACK:
            before_my_board_iterator = chain.from_iterable(before_state.board.black)
            before_opponent_board_iterator = chain.from_iterable(before_state.board.white)
            after_my_board_iterator = chain.from_iterable(after_state.board.black)
            after_opponent_board_iterator = chain.from_iterable(after_state.board.white)
        else:
            before_my_board_iterator = chain.from_iterable(before_state.board.white)
            before_opponent_board_iterator = chain.from_iterable(before_state.board.black)
            after_my_board_iterator = chain.from_iterable(after_state.board.white)
            after_opponent_board_iterator = chain.from_iterable(after_state.board.black)

            
        input_ids = torch.tensor(
            [
                self.get_token(me, op)
                for me, op in zip(
                        before_my_board_iterator,
                        before_opponent_board_iterator
                )
            ],
            dtype=torch.long,
        )
        input_ids[HEIGH * before_state.move.y + before_state.move.x] = OthelloToken.PUT
        
        labels = torch.tensor(
            [
                self.get_token(me, op)
                for me, op in zip(
                        after_my_board_iterator,
                        after_opponent_board_iterator
                )
            ],
            dtype=torch.long,
        )
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(labels, dtype=torch.long),
        }

        

class OthelloMaskedPretrainTokenizer(OthelloTokenizerBase):
    def __init__(self):
        super().__init__()
        self._pad2d = torch.nn.ZeroPad2d(1)
        self._conv2d = torch.nn.functional.conv2d
        self._filter = torch.ones((1, 1, 3, 3), dtype=torch.float32)

    @torch.no_grad()
    def __call__(self, state:OthelloState):
        HEIGH, WIDTH = state.board.size()
        my_color = state.turn
        
        if my_color == Color.BLACK:
            my_board_iterator = chain.from_iterable(state.board.black)
            opponent_board_iterator = chain.from_iterable(state.board.white)
        else:
            my_board_iterator = chain.from_iterable(state.board.white)
            opponent_board_iterator = chain.from_iterable(state.board.black)


        black_bit_tensor = torch.tensor(
            state.board.black,
            dtype=torch.bool
        )
        white_bit_tensor = torch.tensor(
            state.board.white,
            dtype=torch.bool
        )

        
        disk_placed_bit_tensor = torch.logical_or(
            black_bit_tensor,
            white_bit_tensor
        ).float()

        
        conved_tensor = self._conv2d(
            self._pad2d(disk_placed_bit_tensor).unsqueeze(0).unsqueeze(0),
            self._filter,
        ).squeeze().long()
        
        adjacent_bit_tensor = torch.where(
            conved_tensor > 0,
            torch.ones_like(conved_tensor, dtype=torch.bool),
            torch.zeros_like(conved_tensor, dtype=torch.bool),
        )

        mask_bit_tensor = torch.logical_and(
            adjacent_bit_tensor,
            torch.logical_not(disk_placed_bit_tensor)
        )

        legal_move_bit_tensor = torch.tensor(
            state.legal_move_board,
            dtype=torch.bool,
        )

        
        board_tensor = torch.tensor(
            [
                self.get_token(me, op)
                for me, op in zip(
                        my_board_iterator,
                        opponent_board_iterator
                )
            ],
            dtype=torch.long,
        ).view(HEIGH, WIDTH)
        
        input_ids = torch.where(
            mask_bit_tensor,
            torch.full_like(board_tensor, OthelloToken.MASK, dtype=torch.long),
            board_tensor
        ).view(-1)
        
        labels = torch.where(
            legal_move_bit_tensor,
            torch.full_like(board_tensor, OthelloToken.LEAGAL, dtype=torch.long),
            board_tensor
        ).view(-1)

        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
        }





class OthelloEvalScoreRegrssionTokenizer(OthelloTokenizerBase):
    def __init__(self):
        super().__init__()
        self.BLANK_SCORE = -100

    @torch.no_grad()
    def __call__(self, state:OthelloState):
        HEIGH, WIDTH = state.board.size()
        my_color = state.turn
        
        if my_color == Color.BLACK:
            my_board_iterator = chain.from_iterable(state.board.black)
            opponent_board_iterator = chain.from_iterable(state.board.white)
        else:
            my_board_iterator = chain.from_iterable(state.board.white)
            opponent_board_iterator = chain.from_iterable(state.board.black)
            

        legal_move_bit_tensor = torch.tensor(
            state.legal_move_board,
            dtype=torch.bool,
        ).view(-1)

        board_tensor = torch.tensor(
            [
                self.get_token(me, op)
                for me, op in zip(
                        my_board_iterator,
                        opponent_board_iterator
                )
            ],
            dtype=torch.long,
        )

        
        input_ids = torch.where(
            legal_move_bit_tensor,
            torch.full_like(
                legal_move_bit_tensor,
                OthelloToken.LEAGAL,
                dtype=torch.long
            ),
            board_tensor
        )


        labels = torch.tensor(
            [
                s if s is not None else self.BLANK_SCORE
                for s in chain.from_iterable(state.score_board)
            ],
            dtype=torch.float32,
        )
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

class OthelloEvalScoreRegrssionWithAttensionMaskTokenizer(OthelloEvalScoreRegrssionTokenizer):
    
    @torch.no_grad()
    def __call__(self, state:OthelloState):
        HEIGH, WIDTH = state.board.size()
        my_color = state.turn
        
        if my_color == Color.BLACK:
            my_board_iterator = chain.from_iterable(state.board.black)
            opponent_board_iterator = chain.from_iterable(state.board.white)
        else:
            my_board_iterator = chain.from_iterable(state.board.white)
            opponent_board_iterator = chain.from_iterable(state.board.black)
            

        legal_move_bit_tensor = torch.tensor(
            state.legal_move_board,
            dtype=torch.bool,
        ).view(-1)

        board_tensor = torch.tensor(
            [
                self.get_token(me, op)
                for me, op in zip(
                        my_board_iterator,
                        opponent_board_iterator
                )
            ],
            dtype=torch.long,
        )

        
        input_ids = torch.where(
            legal_move_bit_tensor,
            torch.full_like(
                legal_move_bit_tensor,
                OthelloToken.LEAGAL,
                dtype=torch.long
            ),
            board_tensor
        )
        
        labels = torch.tensor(
            [
                s if s is not None else self.BLANK_SCORE
                for s in chain.from_iterable(state.score_board)
            ],
            dtype=torch.float32,
        )
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": legal_move_bit_tensor.long(),
        }
