import argparse
import os
from pathlib import Path
from typing import List, Dict, Union, Optional
from ctypes import *


dylib_path = os.path.join(os.path.dirname(__file__), "_libedax.so")
print(dylib_path)
INSTANCE = cdll.LoadLibrary(dylib_path);
"""
dylib_path = os.path.join(os.path.dirname(__file__), "libedax.so")
LIB = cdll.LoadLibrary(dylib_path)
LIB.libedax_initialize.restype = None;
"""

INSTANCE.libedax_initialize.restype = None;
INSTANCE.libedax_initialize.argtypes = [c_int, POINTER(c_char_p)];
