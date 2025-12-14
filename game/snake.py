import pygame
import random
from typing import List, Tuple


class Snake:
    """
    Represents the snake in the game.
    Handles snake body, movement, growth, and collision detection with itself.
    """
    
    def __init__(self, grid_size: int, block_size: int):
        """
        Initialize the snake at the center of the grid.
        
        Args:
            grid_size: Number of blocks in the grid (e.g., 20)
            block_size: Size of each block in pixels
        """
        self.grid_size = grid_size
        self.block_size = block_size
        
        # Start snake at center of grid
        center = grid_size // 2
        self.body = [
            (center, center),
            (center - 1, center),
            (center - 2, center)
        ]
        
        # Initial direction: moving right
        self.direction = (1, 0)
        self.next_direction = (1, 0)
        
        # Color
        self.head_color = (0, 200, 0)
        self.body_color = (0, 150, 0)
    
    def update_direction(self, new_direction: Tuple[int, int]):
        """
        Update the direction the snake will move.
        Prevents the snake from reversing into itself.
        
        Args:
            new_direction: Tuple (dx, dy) representing direction
        """
        # Prevent reversing direction
        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            self.next_direction = new_direction
    
    def move(self, grow: bool = False):
        """
        Move the snake one step in the current direction.
        
        Args:
            grow: If True, snake grows by one block (after eating fruit)
        """
        self.direction = self.next_direction
        
        # Calculate new head position
        head_x, head_y = self.body[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        
        # Add new head
        self.body.insert(0, new_head)
        
        # Remove tail unless growing
        if not grow:
            self.body.pop()
    
    def check_wall_collision(self) -> bool:
        """
        Check if snake's head has hit a wall.
        
        Returns:
            True if collision with wall detected
        """
        head_x, head_y = self.body[0]
        return (head_x < 0 or head_x >= self.grid_size or
                head_y < 0 or head_y >= self.grid_size)
    
    def check_self_collision(self) -> bool:
        """
        Check if snake's head has collided with its own body.
        
        Returns:
            True if collision with self detected
        """
        head = self.body[0]
        return head in self.body[1:]
    
    def check_collision(self, point: Tuple[int, int]) -> bool:
        """
        Check if a point would collide with walls or snake body.
        
        Args:
            point: Tuple (x, y) to check
        
        Returns:
            True if collision detected
        """
        x, y = point
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        if point in self.body:
            return True
        return False
    
    def get_head_position(self) -> Tuple[int, int]:
        """Get the current position of the snake's head."""
        return self.body[0]
    
    def draw(self, screen: pygame.Surface):
        """
        Draw the snake on the screen.
        
        Args:
            screen: Pygame surface to draw on
        """
        for i, (x, y) in enumerate(self.body):
            rect = pygame.Rect(
                x * self.block_size,
                y * self.block_size,
                self.block_size,
                self.block_size
            )
            
            # Head is slightly different color
            color = self.head_color if i == 0 else self.body_color
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)  # Border

