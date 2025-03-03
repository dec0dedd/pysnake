from enum import Enum
import pygame


class Directions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Colors(Enum):
    RED = pygame.Color(255, 0, 0)
    BLACK = pygame.Color(0, 0, 0)
    WHITE = pygame.Color(255, 255, 255)
    GREEN = pygame.Color(0, 255, 0)
    DARK_GREEN = pygame.Color(33, 117, 38)
    BLUE = pygame.Color(0, 0, 255)
