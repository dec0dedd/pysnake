import pygame
import sys
from typing import Tuple
import os

from game_env import SnakeEnv
from const import Directions, Colors
from ppo_agent import PPOAgent

import numpy as np

TILE_SIZE = 20
SAVE_FRAMES = False

window_x = 720
window_y = 480
assert window_x % TILE_SIZE == 0 and window_y % TILE_SIZE == 0

env = SnakeEnv()
obs, info = env.reset()
direction = Directions.DOWN

errors = pygame.init()
pygame.display.set_caption('Snake')
game_window = pygame.display.set_mode((env.grid_size[0] - 2 * env.tile_size, env.grid_size[1] - 2 * env.tile_size))
fps_controller = pygame.time.Clock()

agent = PPOAgent(env.block_size, "models/ppo/model.zip")

frame_count = 0

if SAVE_FRAMES:
    os.mkdir("frames")


def block2cord(x: int, y: int, tile_size: int) -> Tuple[int, int]:
    return ((x-1)*tile_size, (y-1)*tile_size)


def write_score(score):
    font = pygame.font.Font("fonts/ps2p/PressStart2P.ttf", 20)
    score_surface = font.render("Score: " + str(score), True, Colors.WHITE.value)
    score_rect = score_surface.get_rect()
    score_rect.topleft = (10, 10)
    game_window.blit(score_surface, score_rect)


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))

    direction = agent.act(obs)
    obs, reward, term, trunc, info = env.step(direction)

    if term:
        pygame.quit()
        sys.exit()

    game_window.fill(Colors.BLACK.value)
    for x, y in np.argwhere(info['snake'] == 1):
        pygame.draw.rect(
            game_window,
            Colors.GREEN.value,
            pygame.Rect(*block2cord(x, y, env.tile_size), env.tile_size, env.tile_size)
        )

    for x, y in np.argwhere(info['snake'] == 2):
        pygame.draw.rect(
            game_window,
            Colors.DARK_GREEN.value,
            pygame.Rect(*block2cord(x, y, env.tile_size), env.tile_size, env.tile_size)
        )

    assert len(np.argwhere(info['apple'] == 1)) == 1
    apple_cords = np.argwhere(info['apple'] == 1)[0]
    pygame.draw.rect(
        game_window,
        Colors.RED.value,
        pygame.Rect(*block2cord(apple_cords[0], apple_cords[1], env.tile_size), env.tile_size, env.tile_size)
    )

    for x in range(1, env.block_size[0] + 1):
        for y in range(1, env.block_size[1] + 1):
            pygame.draw.rect(
                game_window,
                Colors.BLACK.value,
                pygame.Rect(*block2cord(x, y, env.tile_size), env.tile_size, env.tile_size),
                width=1
            )

    write_score(info['score'][0])
    pygame.display.update()
    fps_controller.tick(10)

    if SAVE_FRAMES:
        frame_path = os.path.join("frames", f"frame_{frame_count:04d}.png")
        pygame.image.save(game_window, frame_path)
        frame_count += 1
