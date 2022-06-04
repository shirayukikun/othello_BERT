from ctypes import *
from sys import platform as _platform
import os
from pathlib import Path
import shutil
import tempfile

from .libedax import *
from .edaxutil import to_cstr_array 

DEFAULT_BOOK_DATA_PATH = Path(__file__).parent / "../data/book.dat"
DEFAULT_EVAL_DATA_PATH = Path(__file__).parent / "../data/eval.dat"

module_name = ""
if _platform == "linux" or _platform == "linux2":
    module_name = "libedax.so";
elif _platform == "darwin":
    module_name = "libedax.dylib";
elif _platform == "win32":
    module_name = "libedax-x86.dll";
elif _platform == "win64":
    module_name = "libedax-x64.dll";

dylib_path = os.path.join(os.path.dirname(__file__), "_libedax.so")


class Edax:
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = Path(temp_dir.name)
    
    def __init__(
            self,
            book_file_path=DEFAULT_BOOK_DATA_PATH,
            eval_file_path=DEFAULT_EVAL_DATA_PATH,
            level=0,      
    ):
        
        self.temp_dylib_path = self.__class__.temp_dir_path / f"libedax_{id(self)}.so"
        shutil.copy(dylib_path, self.temp_dylib_path)
        self._instance = cdll.LoadLibrary(self.temp_dylib_path)
        self.initialize_instance()
        
        self.edax_args = [
            "",
            "-book-file",
            str(book_file_path),
            "-eval-file",
            str(eval_file_path),
            "-level",
            str(level)
        ]
        
        self.libedax_initialize(
            len(self.edax_args),
            to_cstr_array(self.edax_args)
        )
        self.edax_init()


    def terminate(self):
        self.libedax_terminate()
        self.ui_free_libedax()

    def __dell__(self):
        self.terminate()
        
    # 属性が定義されていなければ, edaxの共有ライブラリのインスタンスを呼び出す
    def __getattr__(self, name):
        return getattr(self._instance, name)


    def initialize_instance(self):
        self._instance.libedax_initialize.restype = None;
        self._instance.libedax_initialize.argtypes = [c_int, POINTER(c_char_p)];

        self._instance.libedax_terminate.restype = None;
        self._instance.libedax_terminate.argtypes = [];

        self._instance.edax_init.restype = None;
        self._instance.edax_init.argtypes = [];

        self._instance.edax_new.restype = None;
        self._instance.edax_new.argtypes = [];

        self._instance.edax_load.restype = None;
        self._instance.edax_load.argtypes = [c_char_p];

        self._instance.edax_save.restype = None;
        self._instance.edax_save.argtypes = [c_char_p];

        self._instance.edax_undo.restype = None;
        self._instance.edax_undo.argtypes = [];

        self._instance.edax_redo.restype = None;
        self._instance.edax_redo.argtypes = [];

        self._instance.edax_mode.restype = None;
        self._instance.edax_mode.argtypes = [c_int];

        self._instance.edax_setboard.restype = None;
        self._instance.edax_setboard.argtypes = [c_char_p];

        self._instance.edax_setboard_from_obj.restype = None;
        self._instance.edax_setboard_from_obj.argtypes = [POINTER(Board), c_int];

        self._instance.edax_vmirror.restype = None;
        self._instance.edax_vmirror.argtypes = [];

        self._instance.edax_hmirror.restype = None;
        self._instance.edax_hmirror.argtypes = [];

        self._instance.edax_rotate.restype = None;
        self._instance.edax_rotate.argtypes = [c_int];

        self._instance.edax_symetry.restype = None;
        self._instance.edax_symetry.argtypes = [c_int];

        self._instance.edax_play.restype = None;
        self._instance.edax_play.argtypes = [c_char_p];

        self._instance.edax_force.restype = None;
        self._instance.edax_force.argtypes = [c_char_p];

        self._instance.edax_go.restype = None;
        self._instance.edax_go.argtypes = [];

        self._instance.edax_hint.restype = None;
        self._instance.edax_hint.argtypes = [c_int, POINTER(HintList)];

        self._instance.edax_hint_prepare.restype = None;
        self._instance.edax_hint_prepare.argtypes = [];

        self._instance.edax_hint_next.restype = None;
        self._instance.edax_hint_next.argtypes = [POINTER(Hint)];

        self._instance.edax_stop.restype = None;
        self._instance.edax_stop.argtypes = [];

        self._instance.edax_move.restype = c_bool;
        self._instance.edax_move.argtypes = [c_char_p];

        self._instance.edax_opening.restype = c_char_p;
        self._instance.edax_opening.argtypes = [];

        self._instance.edax_ouverture.restype = c_char_p;
        self._instance.edax_ouverture.argtypes = [];

        self._instance.edax_book_store.restype = None;
        self._instance.edax_book_store.argtypes = [];

        self._instance.edax_book_on.restype = None;
        self._instance.edax_book_on.argtypes = [];

        self._instance.edax_book_off.restype = None;
        self._instance.edax_book_off.argtypes = [];

        self._instance.edax_book_randomness.restype = None;
        self._instance.edax_book_randomness.argtypes = [c_int];

        self._instance.edax_book_depth.restype = None;
        self._instance.edax_book_depth.argtypes = [c_int];

        self._instance.edax_book_new.restype = None;
        self._instance.edax_book_new.argtypes = [c_int, c_int];

        self._instance.edax_book_load.restype = None;
        self._instance.edax_book_load.argtypes = [c_char_p];

        self._instance.edax_book_save.restype = None;
        self._instance.edax_book_save.argtypes = [c_char_p];

        self._instance.edax_book_import.restype = None;
        self._instance.edax_book_import.argtypes = [c_char_p];

        self._instance.edax_book_export.restype = None;
        self._instance.edax_book_export.argtypes = [c_char_p];

        self._instance.edax_book_merge.restype = None;
        self._instance.edax_book_merge.argtypes = [c_char_p];

        self._instance.edax_book_fix.restype = None;
        self._instance.edax_book_fix.argtypes = [];

        self._instance.edax_book_negamax.restype = None;
        self._instance.edax_book_negamax.argtypes = [];

        self._instance.edax_book_correct.restype = None;
        self._instance.edax_book_correct.argtypes = [];

        self._instance.edax_book_prune.restype = None;
        self._instance.edax_book_prune.argtypes = [];

        self._instance.edax_book_subtree.restype = None;
        self._instance.edax_book_subtree.argtypes = [];

        self._instance.edax_book_show.restype = None;
        self._instance.edax_book_show.argtypes = [POINTER(Position)];

        self._instance.edax_book_info.restype = None;
        self._instance.edax_book_info.argtypes = [POINTER(Book)];

        self._instance.edax_book_verbose.restype = None;
        self._instance.edax_book_verbose.argtypes = [c_int];

        self._instance.edax_book_add.restype = None;
        self._instance.edax_book_add.argtypes = [c_char_p];

        self._instance.edax_book_check.restype = None;
        self._instance.edax_book_check.argtypes = [c_char_p];

        self._instance.edax_book_extract.restype = None;
        self._instance.edax_book_extract.argtypes = [c_char_p];

        self._instance.edax_book_deviate.restype = None;
        self._instance.edax_book_deviate.argtypes = [c_int, c_int];

        self._instance.edax_book_enhance.restype = None;
        self._instance.edax_book_enhance.argtypes = [c_int, c_int];

        self._instance.edax_book_fill.restype = None;
        self._instance.edax_book_fill.argtypes = [c_int];

        self._instance.edax_book_play.restype = None;
        self._instance.edax_book_play.argtypes = [];

        self._instance.edax_book_deepen.restype = None;
        self._instance.edax_book_deepen.argtypes = [];

        self._instance.edax_book_feed_hash.restype = None;
        self._instance.edax_book_feed_hash.argtypes = [];

        self._instance.edax_base_problem.restype = None;
        self._instance.edax_base_problem.argtypes = [c_char_p, c_int, c_char_p];

        self._instance.edax_base_tofen.restype = None;
        self._instance.edax_base_tofen.argtypes = [c_char_p, c_int, c_char_p];

        self._instance.edax_base_correct.restype = None;
        self._instance.edax_base_correct.argtypes = [c_char_p, c_int];

        self._instance.edax_base_complete.restype = None;
        self._instance.edax_base_complete.argtypes = [c_char_p];

        self._instance.edax_base_convert.restype = None;
        self._instance.edax_base_convert.argtypes = [c_char_p, c_char_p];

        self._instance.edax_base_unique.restype = None;
        self._instance.edax_base_unique.argtypes = [c_char_p, c_char_p];

        self._instance.edax_set_option.restype = None;
        self._instance.edax_set_option.argtypes = [c_char_p, c_char_p];

        self._instance.edax_get_moves.restype = c_char_p;
        self._instance.edax_get_moves.argtypes = [c_char_p];

        self._instance.edax_is_game_over.restype = c_bool;
        self._instance.edax_is_game_over.argtypes = [];

        self._instance.edax_can_move.restype = c_bool;
        self._instance.edax_can_move.argtypes = [];

        self._instance.edax_get_last_move.restype = None;
        self._instance.edax_get_last_move.argtypes = [POINTER(Move)];

        self._instance.edax_get_board.restype = None;
        self._instance.edax_get_board.argtypes = [POINTER(Board)];

        self._instance.edax_get_current_player.restype = c_int;
        self._instance.edax_get_current_player.argtypes = [];

        self._instance.edax_get_disc.restype = c_int;
        self._instance.edax_get_disc.argtypes = [c_int];

        self._instance.edax_get_mobility_count.restype = c_int;
        self._instance.edax_get_mobility_count.argtypes = [c_int];

        ### board.c function

        self._instance.get_moves.restype = c_ulonglong;
        self._instance.get_moves.argtypes = [c_ulonglong, c_ulonglong];

        self._instance.can_move.restype = c_bool;
        self._instance.can_move.argtypes = [c_ulonglong, c_ulonglong];

        ### bit.c function

        self._instance.bit_count.restype = c_int;
        self._instance.bit_count.argtypes = [c_ulonglong];

        self._instance.first_bit.restype = c_int;
        self._instance.first_bit.argtypes = [c_ulonglong];

        self._instance.last_bit.restype = c_int;
        self._instance.last_bit.argtypes = [c_ulonglong];

