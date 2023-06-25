import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import pygame
import numpy as np
import matplotlib.pyplot as plt

from sympy import true
from network import Linear_QNet2
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE

random.seed(9001)

class Agent:
    def __init__(self):
        self.gamma = 0.9  # poziom "zniżki"
        self.epsilon = 0  # losowość
        self.n_game = 0
        self.memory = deque()
        self.model = Linear_QNet2(15, 256, 3)
        self.model.train()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # niebezpieczeństwo przed
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_r)),

            # niebezpieczenstwo prawo
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # niebezpieczenstwo lewo
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # kierunek ruchu
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # pozycja jedzenia wzgledem glowy
            game.food.x < game.head.x,  # jedzenie po lewej
            game.food.x > game.head.x,  # jedzenie po prawej
            game.food.y < game.head.y,  # jedzenie nad
            game.food.y > game.head.y,   # jedzenie za
            # pozycja super jedzenia wzgledem glowy
            game.food.x < game.head.x,  # super jedzenie po lewej
            game.superFood.x > game.head.x,  # super jedzenie po prawej
            game.superFood.y < game.head.y,  # super jedzenie nad
            game.superFood.y > game.head.y  # super jedzenie pod

        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])  # popleft if memory exceed
        if len(self.memory) > 10000:
            self.memory.popleft()

    def train_long_memory(self, memory):
        self.n_game += 1
        mini_sample = memory

        state, action, reward, next_state, done = zip(*mini_sample)
        state = torch.tensor(np.array(state), dtype=torch.float)  # [1, ... , 0]
        action = torch.tensor(action, dtype=torch.long)  # [1, 0, 0]
        reward = torch.tensor(reward, dtype=torch.float)  # int
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)  # [True, ... , False]
        target = reward + self.gamma * torch.max(self.model(next_state), dim=1)[0]
        location = [[x] for x in torch.argmax(action, dim=1).numpy()]
        location = torch.tensor(location)
        pred = self.model(state).gather(1, location)
        pred = pred.squeeze(1)
        loss = self.loss_fn(target, pred)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_short_memory(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        target = reward
        if not done:
            target = reward + self.gamma * torch.max(self.model(next_state))
        pred = self.model(state)
        target_f = pred.clone()
        target_f[torch.argmax(action).item()] = target
        loss = self.loss_fn(target_f, pred)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def plot(self, score, mean_per_game):
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(score)
        plt.plot(mean_per_game)
        plt.ylim(ymin=0)
        plt.text(len(score) - 1, score[-1], str(score[-1]))
        plt.text(len(mean_per_game) - 1, mean_per_game[-1], str(mean_per_game[-1]))
        plt.pause(0.001)

    def get_action(self, state):
        self.epsilon = 100 - self.n_game
        final_move = [0, 0, 0]
        if random.randint(0, 250) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(np.array(state), dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


class DQNAgent_play(object):

    def __init__(self, path):
        self.counter_games = 0
        self.model = Linear_QNet2(15, 256, 3)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # niebezpieczeństwo przed
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_r)),

            # niebezpieczenstwo prawo
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # niebezpieczenstwo lewo
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # kierunek ruchu
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # pozycja jedzenia wzgledem glowy
            game.food.x < game.head.x,  # jedzenie po lewej
            game.food.x > game.head.x,  # jedzenie po prawej
            game.food.y < game.head.y,  # jedzenie nad
            game.food.y > game.head.y,  # jedzenie za
            # pozycja super jedzenia wzgledem glowy
            game.food.x < game.head.x,  # super jedzenie po lewej
            game.superFood.x > game.head.x,  # super jedzenie po prawej
            game.superFood.y < game.head.y,  # super jedzenie nad
            game.superFood.y > game.head.y  # super jedzenie pod

        ]
        return np.array(state, dtype=int)

    def plot(self, score, mean_per_game):
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(score)
        plt.plot(mean_per_game)
        plt.ylim(ymin=0)
        plt.text(len(score) - 1, score[-1], str(score[-1]))
        plt.text(len(mean_per_game) - 1, mean_per_game[-1], str(mean_per_game[-1]))
        plt.pause(0.001)

    def get_action(self, state):
        final_move = [0, 0, 0]
        state0 = torch.tensor(np.array(state), dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] += 1
        return final_move


def train():

    model_folder_path = './model'
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
    plt.ion()
    pygame.init()
    plot_scores = []
    total_score = 0
    plot_mean_scores = []
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)
        if done == true:
            game.reset()
            agent.n_game += 1
            agent.train_long_memory(agent.memory)
            if score > record:
                record = score
                torch.save(agent.model.state_dict(), 'model.pth')
            print('Game:', agent.n_game / 2, 'Score:', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_game
            plot_mean_scores.append(mean_score)
            agent.plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()