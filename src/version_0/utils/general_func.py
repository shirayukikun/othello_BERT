from itertools import islice, tee
from collections import deque

# Copyed from toolz
def sliding_window(seq, n):
    return zip(*(deque(islice(it, i), 0) or it for i, it in enumerate(tee(seq, n))))
