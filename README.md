# Snake RL Project

A Deep Reinforcement Learning project that trains an AI agent to play Snake using Deep Q-Learning (DQN).

## Project Structure

```
Snake-RL-Project/
│
├── game/
│   ├── game.py                # Game window management, score, game loop
│   ├── snake.py               # Snake body, movement, collisions
│
├── model/
│   ├── dqn_agent.py           # RL agent (future step)
│   ├── model.py               # Neural network (future step)
│
├── utils/
│   ├── plot.py                # Plotting scores (future step)
│
├── results/
│   ├── score_history.csv      # Save results (future step)
│
├── main.py                    # Entry point to run environment
└── README.md


### Features

- 600x600 Pygame window with 20x20 grid
- Snake movement with arrow keys or WASD
- Random fruit spawning
- Collision detection (walls and self)
- Score tracking
- Game over and restart functionality
- Smooth game loop

## Installation

1. Install required dependencies:

```bash
pip install pygame torch
```

## Running the Game

To play the game manually:

```bash
python main.py
```

### Controls

- **Arrow Keys** or **WASD**: Control snake direction
- **SPACE**: Restart after game over




