import pygame
from enum import Enum
from collections import namedtuple
import numpy as np
import time

pygame.init()
font_size = 24
font = pygame.font.Font(None, font_size)

Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 20
SPEED = 1000
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

x_array = np.array([12, 8, 3, 14, 9, 5, 1, 6, 7, 10, 15, 0, 13, 4, 2, 11, 7, 10, 8, 3])
y_array = np.array([3, 2, 5, 7, 4, 1, 0, 6, 4, 8, 10, 3, 1, 9, 2, 7, 10, 5, 0, 3])

x_array_2 = np.array([9, 2, 13, 7, 4, 11, 0, 6, 14, 8, 15, 3, 10, 9, 12, 1, 14, 5, 0, 11])
y_array_2 = np.array([9, 2, 5, 7, 4, 1, 0, 6, 4, 8, 10, 3, 1, 9, 2, 7, 10, 5, 0, 3])


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class SnakeGameAI:
    def __init__(self, w=320, h=240):
        self.action = None
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        self.frame_iteration = 0

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.reward = 0
        self.food = None
        self.superFood = None
        self.food_index = 0
        self.superFood_index = 0
        self._place__food()

        # Initial game start time and max game time in seconds
        self.game_start_time = time.time()
        self.max_game_time = 30  # Max game time in seconds
        self.eat_timer = 3

        # time variables
        self.start_time = time.time()
        self.elapsed_time = 0

    def _place__food(self):

        x1 = x_array[self.food_index] * 20
        y1 = y_array[self.food_index] * 20
        self.food = Point(x1, y1)
        x2 = x_array_2[self.superFood_index] * 20
        y2 = y_array_2[self.superFood_index] * 20
        self.superFood = Point(x2, y2)

        if self.food in self.snake:
            self.score += 1
            self.food_index += 1
            self.eat_timer += 2
            #self.snake.append(self.food)
            self._place__food()

        if self.superFood in self.snake:
            self.score += 2
            self.superFood_index += 1
            self.eat_timer += 2
            #self.snake.append(self.superFood)
            self._place__food()

    def play_step(self, action):
        self.frame_iteration += 1
        # Collect the user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()



        # Check if total game time exceeded
        elapsed_time = time.time() - self.game_start_time
        if elapsed_time > self.max_game_time:
            print('lol1')
            self.reward += 100
            return self.reward, True, self.score
        # Check if total game time or eat time
        elapsed_time = time.time() - self.game_start_time  # czars triennial gry
        if elapsed_time > self.max_game_time or elapsed_time > self.eat_timer:
            self.reward -= 50
            print('lol2')
            return self.reward, True, self.score

        # Move
        self._move(action)
        self.snake.insert(0, self.head)

        # Check if game over
        game_over = False
        if self.is_collision():
            game_over = True
            self.reward += -15
            print('lol3')
            return self.reward, game_over, self.score

        # Place new food or just move
        if self.head == self.food:
            self.score += 1
            self.reward += 15
            self.food_index = (self.food_index + 1)
            self.snake.append(self.food)
            self.eat_timer += 2
            self._place__food()
        elif self.head == self.superFood:
            self.score += 2
            self.reward += 25
            self.superFood_index = (self.superFood_index + 1)
            self.snake.append(self.superFood)
            self.eat_timer += 2
            self._place__food()
        else:
            self.snake.pop()

        # Update UI and clock
        self._update_ui(elapsed_time)
        self.clock.tick(SPEED)

        # Calculate elapsed time
        self.elapsed_time = time.time() - self.start_time

        # Check if 5 seconds have passed
        if self.elapsed_time > self.max_game_time or elapsed_time > self.eat_timer:
            game_over = True
            print('lol4')
            return game_over, self.score, self.reward

        # Return game over and display score
        return False, self.score, self.reward

    def _update_ui(self, elapsed_time):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.superFood.x, self.superFood.y, BLOCK_SIZE, BLOCK_SIZE))

        score_text = "Score: " + str(self.score)
        score_surface = font.render(score_text, True, WHITE)
        self.display.blit(score_surface, [0, 0])

        remaining_game_time = self.max_game_time - int(elapsed_time)
        game_time_text = "Total game time left: " + str(remaining_game_time)
        game_time_surface = font.render(game_time_text, True, WHITE)
        self.display.blit(game_time_surface, [0, 30])

        remaining_eat_time = self.eat_timer - int(elapsed_time)
        eat_time_text = "Food time left: " + str(remaining_eat_time)
        eat_time_surface = font.render(eat_time_text, True, WHITE)
        self.display.blit(eat_time_surface, [0, 60])

        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1}
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    def is_collision(self, point=None):
        if point is None:
            point = self.head
        if (
                point.x > self.w - BLOCK_SIZE
                or point.x < 0
                or point.y > self.h - BLOCK_SIZE
                or point.y < 0
        ):
            return True
        return False
