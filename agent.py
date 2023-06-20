import torch
import torch.nn as nn
import torch.optim as optim

from network import Linear_QNet2
import random
import os
import pygame
import numpy as np
from model import Linear_QNet, QTrainer
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
import matplotlib.pyplot as plt

random.seed(9001)


class Agent:

    def __init__(self):
        self.gamma = 0.9  # discount rate
        self.epsilon = 0  # Randomness
        self.n_game = 0
        self.memory = deque()  # popleft()

        self.model = Linear_QNet2(7, 256, 3)
        self.model.train()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
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
            # Danger Straight
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_r)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger Left
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])  # popleft if memory exceed
        if len(self.memory) > 100000:
            self.memory.popleft()

    def train_long_memory(self, memory):
        self.n_game += 1
        if len(memory) > 1000:
            mini_sample = random.sample(memory, 1000)
        else:
            mini_sample = memory

        state, action, reward, next_state, done = zip(*mini_sample)
        state = torch.tensor(state, dtype=torch.float)  # [1, ... , 0]
        action = torch.tensor(action, dtype=torch.long)  # [1, 0, 0]
        reward = torch.tensor(reward, dtype=torch.float)  # int
        next_state = torch.tensor(next_state, dtype=torch.float)  # [True, ... , False]
        target = reward
        target = reward + self.gamma * torch.max(self.model(next_state), dim=1)[0]
        location = [[x] for x in torch.argmax(action, dim=1).numpy()]
        location = torch.tensor(location)
        pred = self.model(state).gather(1, location)  # [action]
        pred = pred.squeeze(1)
        loss = self.loss_fn(target, pred)
        loss.backward()
        self.optimizer.step()

    def train_short_memory(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
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
        from IPython import display
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(score)
        plt.plot(mean_per_game)
        plt.ylim(ymin=0)
        plt.text(len(score) - 1, score[-1], str(score[-1]))
        plt.text(len(mean_per_game) - 1, mean_per_game[-1], str(mean_per_game[-1]))

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 250 - self.n_game
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        print(final_move)
        return final_move


class DQNAgent_play(object):

    def __init__(self, path):
        self.counter_games = 0
        self.model = Linear_QNet2(11, 256, 3)
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
            # Danger Straight
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_r)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger Left
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

        ]
        return np.array(state, dtype=int)

    def plot(self, score, mean_per_game):
        from IPython import display
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(score)
        plt.plot(mean_per_game)
        plt.ylim(ymin=0)
        plt.text(len(score) - 1, score[-1], str(score[-1]))
        plt.text(len(mean_per_game) - 1, mean_per_game[-1], str(mean_per_game[-1]))

    def get_action(self, state):
        final_move = [0, 0, 0]
        state0 = torch.tensor(state, dtype=torch.float)
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
        # Get Old state
        state_old = agent.get_state(game)
        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        if done:
            print('training started')
            # Train long memory,plot result
            game.reset()
            total_score += score
            agent.train_long_memory(agent.memory)
            print('Game', agent.n_game, '      Score:', score)
            if score > record:
                record = score
                name = 'best_model.pth'.format(score)
                dir = os.path.join(model_folder_path, name)
                torch.save(agent.model.state_dict(), dir)
            print('record: ', record)
            plot_scores.append(score)
            mean = total_score / agent.n_game
            plot_mean_scores.append(mean)
            agent.plot(plot_scores, plot_mean_scores)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    pygame.display.set_caption('Deep Q Snake!')
    pygame.init()
    print('before train()')
    train()
