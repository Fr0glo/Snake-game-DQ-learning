"""
Main entry point for the Snake game environment.
Run this file to train the DQN agent.
"""

from game.game import Game
from model.dqn_agent import Agent
from utils.plot import plot
import pygame

def train():
    scores = []
    mean_scores = []
    record = 0
    agent = Agent()
    game = Game(width=600, height=600, grid_size=20, training_mode=True)
    
    while True:
        state_old = agent.get_state(game)
        
        final_move = agent.get_action(state_old)
        
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
            
            scores.append(score)
            mean_score = sum(scores) / len(scores)
            mean_scores.append(mean_score)
            
            print(f"Game {agent.n_games} | Score: {score} | Record: {record}")
            
            plot(scores, mean_scores)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        game._draw()
        game.clock.tick(60)


if __name__ == "__main__":
    train()


