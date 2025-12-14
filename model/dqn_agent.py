import torch
import random
import numpy as np
from collections import deque
from model.model import Linear_QNet, QTrainer

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    def get_state(self, game):
        head = game.snake.get_head_position()
        point_l = (head[0] - 1, head[1])
        point_r = (head[0] + 1, head[1])
        point_u = (head[0], head[1] - 1)
        point_d = (head[0], head[1] + 1)
        
        dir_l = game.snake.direction == (-1, 0)
        dir_r = game.snake.direction == (1, 0)
        dir_u = game.snake.direction == (0, -1)
        dir_d = game.snake.direction == (0, 1)
        
        state = [
            (dir_r and game.snake.check_collision(point_r)) or
            (dir_l and game.snake.check_collision(point_l)) or
            (dir_u and game.snake.check_collision(point_u)) or
            (dir_d and game.snake.check_collision(point_d)),
            
            (dir_u and game.snake.check_collision(point_r)) or
            (dir_d and game.snake.check_collision(point_l)) or
            (dir_l and game.snake.check_collision(point_u)) or
            (dir_r and game.snake.check_collision(point_d)),
            
            (dir_d and game.snake.check_collision(point_r)) or
            (dir_u and game.snake.check_collision(point_l)) or
            (dir_r and game.snake.check_collision(point_u)) or
            (dir_l and game.snake.check_collision(point_d)),
            
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            game.fruit[0] < head[0],
            game.fruit[0] > head[0],
            game.fruit[1] < head[1],
            game.fruit[1] > head[1],
        ]
        
        return np.array(state, dtype=int)
    
    def get_action(self, state):
        self.epsilon = max(0, 80 - self.n_games)
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move.index(1)
    
    def remember(self, state, action, reward, next_state, done):
        action_one_hot = [0, 0, 0]
        action_one_hot[action] = 1
        self.memory.append((state, action_one_hot, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        action_one_hot = [0, 0, 0]
        action_one_hot[action] = 1
        self.trainer.train_step(state, action_one_hot, reward, next_state, done)
