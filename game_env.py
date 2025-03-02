from typing import Tuple, List, Optional, Dict

from const import Directions

import numpy as np
import gymnasium as gym


class SnakeEnv(gym.Env):
    def __init__(
            self,
            size: Tuple[int, int] = (720, 480),
            tile_size: int = 20,
            snake_start: Optional[List[int]] = None,
            start_dir: Directions = Directions.DOWN,
            ):
        assert size[0] % tile_size == 0 and size[1] % tile_size == 0, "Grid dimensions have to be be divisible by tile_size"
        self.grid_size = size
        self.block_size = (self.grid_size[0] // tile_size, self.grid_size[1] // tile_size)
        self.tile_size = tile_size

        self.snake_body = self._gen_snake_start(self.block_size, snake_start)
        self.apple = self._gen_apple(self.block_size)

        self.start_dir = start_dir
        self.direction = self.start_dir
        self.prev_direction = self.start_dir
        self.score = 0

        self.observation_space = gym.spaces.Dict(
            {
                "snake": gym.spaces.Box(
                    low=0, high=2, shape=self.block_size, dtype=np.int8
                ),

                "apple": gym.spaces.Box(
                    low=0, high=1, shape=self.block_size, dtype=np.int8
                ),

                "score": gym.spaces.Box(
                    low=0, high=np.prod(self.block_size), shape=(1,), dtype=np.int16
                )
            }
        )

        self.action_space = gym.spaces.Discrete(4)

    def _gen_snake_start(
            self,
            block_size: Tuple[int, int] = (36, 24),
            snake_head: Optional[List[int]] = None
            ) -> List[int]:

        if snake_head is None:
            snake_body = [
                [np.random.randint(2, block_size[0] - 2), np.random.randint(2, block_size[1] - 2)]
            ]
        else:
            snake_body = [snake_head]

        assert len(snake_body) == 1
        snake_body.append(
            [snake_body[0][0], snake_body[0][1]-1]
        )

        return snake_body

    def _gen_apple(
            self,
            block_size: Tuple[int, int] = (36, 24),
            ) -> List[int]:

        apple = [
            np.random.randint(1, block_size[0] - 1),
            np.random.randint(1, block_size[1] - 1)
        ]

        return apple

    def _get_obs(self) -> Dict:
        snake_np = np.array(self.snake_body)
        snake_matrix = np.zeros(shape=self.block_size, dtype=np.int8)
        snake_matrix[snake_np[:, 0], snake_np[:, 1]] = 1
        snake_matrix[self.snake_body[0][0], self.snake_body[0][1]] = 2

        apple_matrix = np.zeros(shape=self.block_size, dtype=np.int8)
        apple_matrix[self.apple[0], self.apple[1]] = 1

        return {
            "snake": snake_matrix,
            "apple": apple_matrix,
            "score": self.score,
        }

    def _get_info(self):
        return {}

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
            ):

        super().reset(seed=seed)
        np.random.seed(seed=seed)

        self.snake_body = self._gen_snake_start(self.block_size)
        self.apple = self._gen_apple(self.block_size)
        self.score = 0
        self.direction = self.start_dir
        self.prev_direction = self.start_dir

        return self._get_obs(), self._get_info()

    def step(
            self,
            action: int
            ):
        direction = Directions(action)

        assert len(self.snake_body) > 0
        snake_head = self.snake_body[0].copy()
        if direction == Directions.UP:
            snake_head[1] -= 1
        elif direction == Directions.RIGHT:
            snake_head[0] += 1
        elif direction == Directions.DOWN:
            snake_head[1] += 1
        elif direction == Directions.LEFT:
            snake_head[0] -= 1

        self.snake_body.insert(0, snake_head)
        if self.snake_body[0][0] == self.apple[0] and self.snake_body[0][1] == self.apple[1]:
            self.apple = self._gen_apple(self.block_size)
            self.score += 1
        else:
            self.snake_body.pop()

        end_game = False
        if self.snake_body[0][0] < 1 or self.snake_body[0][0] > self.block_size[0] - 2:
            print("Out of bounds on X!")
            end_game = True
        elif self.snake_body[0][1] < 1 or self.snake_body[0][1] > self.block_size[1] - 2:
            print("Out of bounds! on Y!")
            end_game = True

        for block in self.snake_body[1:]:
            if block[0] == self.snake_body[0][0] and block[1] == self.snake_body[0][1]:
                print(f"Clash! {block} - {self.snake_body}")
                end_game = True

        reward = len(self.snake_body)
        if end_game:
            reward = -1

        return self._get_obs(), reward, end_game, False, self._get_info()