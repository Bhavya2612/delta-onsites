import numpy as np
import itertools
import random
from collections import Counter


class env:
    def __init__(self, input_code):
        if isinstance(input_code, int):
            input_code = self._number_from_index(input_code)
        self.input_code = input_code

    def _index_from_number(number):
        assert(len(number) <= 4)
        assert(set(number) <= set(map(str, range(6))))
        return int(number, base=6)
        
    def _number_from_index(index):
        digits = []
        while index > 0:
            digits.append(str(index % 6))
            index = index // 6
        return "".join(reversed(digits)).zfill(4)
    
    def score(p, q):
        right = sum(p_i == q_i for p_i, q_i in zip(p, q))
        misses = sum((Counter(p) & Counter(q)).values()) - right
        return right, misses
    
    def get_feedback(self,action):
        return self.score(self.input_code, action)
    
    def reward(self, guess):
        if guess == self.input_code:
            return 1
        else:
            return -1
         