from typing import List

from const import Directions

import numpy as np


class SimpleAgent:
    def __init__(
            self,
            block_size: List[int]
            ):
        self.block_size = block_size

    def act(self, obs, info):
        head = np.argwhere(info['snake'] == 2)[0]

        move = Directions.DOWN
        if head[0] == 1:
            if head[1] == 1:
                if obs['snake'][head[0] + 1, head[1]] == 0:
                    move = Directions.RIGHT
                elif obs['snake'][head[0], head[1] + 1] == 0:
                    move = Directions.DOWN
            elif head[1] == self.block_size[1] - 2:
                if obs['snake'][head[0], head[1] - 1] == 0:
                    move = Directions.UP
                elif obs['snake'][head[0] + 1, head[1]] == 0:
                    move = Directions.RIGHT
            else:
                if obs['snake'][head[0], head[1] - 1] == 0:
                    move = Directions.UP
                else:
                    move = Directions.DOWN
        elif head[0] == self.block_size[0] - 2:
            if head[1] == 1:
                if obs['snake'][head[0] - 1, head[1]] == 0:
                    move = Directions.LEFT
                elif obs['snake'][head[0], head[1] + 1] == 0:
                    move = Directions.DOWN
            elif head[1] == self.block_size[1] - 2:
                if obs['snake'][head[0] - 1, head[1]] == 0:
                    move = Directions.LEFT
                elif obs['snake'][head[0], head[1] - 1] == 0:
                    move = Directions.UP
            else:
                if obs['snake'][head[0], head[1] - 1] == 0:
                    move = Directions.UP
                elif obs['snake'][head[0], head[1] + 1] == 0:
                    move = Directions.DOWN
        else:
            if obs['snake'][head[0] - 1, head[1]] == 0:
                move = Directions.LEFT
            elif obs['snake'][head[0] + 1, head[1]] == 0:
                move = Directions.RIGHT
        return move.value
