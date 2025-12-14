import pygame
import random
from typing import Tuple, Optional
from game.snake import Snake


class Game:
    """
    Main game class that manages the game window, game loop, score, and fruit.
    """
    
    def __init__(self, width: int = 600, height: int = 600, grid_size: int = 20, training_mode: bool = False):
        """
        Initialize the game.
        
        Args:
            width: Window width in pixels
            height: Window height in pixels
            grid_size: Number of blocks per side (creates grid_size x grid_size grid)
            training_mode: If True, disable keyboard input for RL training
        """
        pygame.init()
        
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.block_size = width // grid_size
        self.training_mode = training_mode
        
        # Create window
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake Game - DQN Training")
        
        # Game state
        self.clock = pygame.time.Clock()
        self.score = 0
        self.game_over = False
        
        # Colors
        self.bg_color = (20, 20, 20)
        self.fruit_color = (200, 50, 50)
        self.text_color = (255, 255, 255)
        
        # Initialize snake and fruit
        self.snake = Snake(self.grid_size, self.block_size)
        self.fruit = self._spawn_fruit()
        
        # Font for score display
        self.font = pygame.font.Font(None, 36)
        self.game_over_font = pygame.font.Font(None, 48)
    
    def _spawn_fruit(self) -> Tuple[int, int]:
        """
        Spawn a fruit at a random position that doesn't overlap with the snake.
        
        Returns:
            Tuple (x, y) representing fruit position on grid
        """
        while True:
            fruit_pos = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            )
            # Make sure fruit doesn't spawn on snake
            if fruit_pos not in self.snake.body:
                return fruit_pos
    
    def _handle_input(self):
        """Handle keyboard input for snake movement."""
        if self.training_mode:
            return
        
        keys = pygame.key.get_pressed()
        
        # Arrow keys or WASD
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.snake.update_direction((0, -1))
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.snake.update_direction((0, 1))
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.snake.update_direction((-1, 0))
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.snake.update_direction((1, 0))
    
    def _update(self):
        """Update game state: move snake, check collisions, check fruit eating."""
        if self.game_over:
            return
        
        # Get current head position before moving
        current_head = self.snake.get_head_position()
        
        # Calculate where head will be after moving (use next_direction which will be applied)
        next_head = (
            current_head[0] + self.snake.next_direction[0],
            current_head[1] + self.snake.next_direction[1]
        )
        
        # Check if snake will eat fruit
        ate_fruit = (next_head == self.fruit)
        
        # Move snake
        if ate_fruit:
            self.score += 1
            self.snake.move(grow=True)
            self.fruit = self._spawn_fruit()
        else:
            self.snake.move(grow=False)
        
        # Check collisions
        if self.snake.check_wall_collision() or self.snake.check_self_collision():
            self.game_over = True
    
    def _draw(self):
        """Draw everything on the screen."""
        # Clear screen
        self.screen.fill(self.bg_color)
        
        # Draw grid lines (optional, for visual clarity)
        for i in range(self.grid_size + 1):
            # Vertical lines
            pygame.draw.line(
                self.screen,
                (40, 40, 40),
                (i * self.block_size, 0),
                (i * self.block_size, self.height)
            )
            # Horizontal lines
            pygame.draw.line(
                self.screen,
                (40, 40, 40),
                (0, i * self.block_size),
                (self.width, i * self.block_size)
            )
        
        # Draw fruit
        fruit_rect = pygame.Rect(
            self.fruit[0] * self.block_size,
            self.fruit[1] * self.block_size,
            self.block_size,
            self.block_size
        )
        pygame.draw.rect(self.screen, self.fruit_color, fruit_rect)
        pygame.draw.rect(self.screen, (150, 0, 0), fruit_rect, 2)
        
        # Draw snake
        self.snake.draw(self.screen)
        
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, self.text_color)
        self.screen.blit(score_text, (10, 10))
        
        # Draw game over message
        if self.game_over:
            game_over_text = self.game_over_font.render("GAME OVER", True, self.text_color)
            restart_text = self.font.render("Press SPACE to restart", True, self.text_color)
            
            # Center the text
            game_over_rect = game_over_text.get_rect(center=(self.width // 2, self.height // 2 - 30))
            restart_rect = restart_text.get_rect(center=(self.width // 2, self.height // 2 + 20))
            
            self.screen.blit(game_over_text, game_over_rect)
            self.screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _action_to_direction(self, action: int) -> Tuple[int, int]:
        """
        Convert action (0=straight, 1=right, 2=left) to direction vector.
        
        Args:
            action: 0 = straight, 1 = right turn, 2 = left turn
        
        Returns:
            Direction tuple (dx, dy)
        """
        current_dir = self.snake.direction
        
        if action == 0:
            return current_dir
        elif action == 1:
            return (-current_dir[1], current_dir[0])
        else:
            return (current_dir[1], -current_dir[0])
    
    def play_step(self, action: int) -> Tuple[int, bool, int]:
        """
        Execute one step of the game for RL training.
        
        Args:
            action: 0 = straight, 1 = right turn, 2 = left turn
        
        Returns:
            Tuple of (reward, done, score)
        """
        if self.game_over:
            score_before_reset = self.score
            self.reset()
            return -10, True, score_before_reset
        
        head_before = self.snake.get_head_position()
        distance_before = self._manhattan_distance(head_before, self.fruit)
        
        new_direction = self._action_to_direction(action)
        self.snake.update_direction(new_direction)
        
        head_after = (
            head_before[0] + new_direction[0],
            head_before[1] + new_direction[1]
        )
        distance_after = self._manhattan_distance(head_after, self.fruit)
        
        ate_fruit = (head_after == self.fruit)
        
        if ate_fruit:
            self.score += 1
            self.snake.move(grow=True)
            self.fruit = self._spawn_fruit()
            reward = 10
        else:
            self.snake.move(grow=False)
            reward = -0.1
            
            if distance_after < distance_before:
                reward += 1
            elif distance_after > distance_before:
                reward -= 1
        
        if self.snake.check_wall_collision() or self.snake.check_self_collision():
            self.game_over = True
            reward = -10
            done = True
        else:
            done = False
        
        score_to_return = self.score
        
        if done:
            self.reset()
        
        return reward, done, score_to_return
    
    def reset(self):
        """Reset the game to initial state."""
        self.score = 0
        self.game_over = False
        self.snake = Snake(self.grid_size, self.block_size)
        self.fruit = self._spawn_fruit()
    
    def run(self, fps: int = 10):
        """
        Main game loop.
        
        Args:
            fps: Frames per second (controls game speed)
        """
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and self.game_over:
                        self.reset()
            
            # Handle continuous input
            self._handle_input()
            
            # Update game state
            self._update()
            
            # Draw everything
            self._draw()
            
            # Control game speed
            self.clock.tick(fps)
        
        pygame.quit()

