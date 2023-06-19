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
SPEED = 20
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

x_array = np.array([5, 10, 15, 20, 25, 30, 0, 4, 8, 12, 16, 19, 23, 27, 31, 3, 7, 11, 14, 18])
y_array = np.array([8, 15, 2, 19, 5, 12, 21, 4, 23, 7, 3, 16, 1, 20, 9, 11, 13, 6, 18, 10])

x_array_2 = np.array([11, 5, 23, 8, 19, 31, 3, 0, 14, 22, 16, 1, 28, 20, 9, 7, 27, 13, 2, 30])
y_array_2 = np.array([3, 16, 20, 12, 6, 19, 9, 5, 2, 18, 10, 14, 21, 8, 1, 7, 13, 11, 4, 15])

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class SnakeGame:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self.superFood = None
        self.food_index = 0
        self.superFood_index = 0
        self._place__food()

        # Initial game start time and max game time in seconds
        self.game_start_time = time.time()
        self.max_game_time = 30  # Max game time in seconds
        self.eat_timer = 5

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
            self.eat_timer += 5
            self.snake.append(self.food)
            self._place__food()

        if self.superFood in self.snake:
            self.score += 1
            self.superFood_index += 1
            self.eat_timer += 5
            self.snake.append(self.superFood)
            self._place__food()

    def play_step(self):
        # Collect the user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN

        # Check if total game time or eat time
        elapsed_time = time.time() - self.game_start_time  # czars triennial gry
        if elapsed_time > self.max_game_time or elapsed_time > self.eat_timer:
            return True, self.score

        # Move
        self._move(self.direction)
        self.snake.insert(0, self.head)

        # Check if game over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score

        # Place new food or just move
        if self.head == self.food:
            self.score += 1
            self.food_index = (self.food_index + 1)
            self.snake.append(self.food)
            self.eat_timer += 5
            self._place__food()
        elif self.head == self.superFood:
            self.score += 1
            self.superFood_index = (self.superFood_index + 1)
            self.snake.append(self.superFood)
            self.eat_timer += 5
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
            return game_over, self.score

        # Return game over and display score
        return game_over, self.score

    def _update_ui(self, elapsed_time):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.superFood.x, self.superFood.y, BLOCK_SIZE, BLOCK_SIZE))

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

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    def _is_collision(self):
        if (
                self.head.x > self.w - BLOCK_SIZE
                or self.head.x < 0
                or self.head.y > self.h - BLOCK_SIZE
                or self.head.y < 0
        ):
            return True
        if self.head in self.snake[1:]:
            return True
        return False


if __name__ == "__main__":
    game = SnakeGame()

    # Game loop
    while True:
        game_over, score = game.play_step()
        if game_over:
            break
    print('Final Score:', score)

    pygame.quit()