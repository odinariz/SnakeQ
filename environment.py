import numpy as np
import random

import snake_sensors

class Environment:
    def __init__(self, row):
        # parameters
        self.board_info = {"empty": 0, "snake": 1, "apple": 2}
        self.Sensors = snake_sensors.SnakeSensors(row, self.board_info) # all logic to get state

        # Generate components
        self.x = row
        self.restart_game()
    
    def generate_grid(self):
        # generate board (grid) of zeros (always square)
        self.board = np.zeros((self.x, self.x))
    
    def generate_snake(self):
        # Randomly choose spot to generate snake
        indices = np.random.randint(0, high=self.x, size=2)
        y, x = indices[0], indices[1]

        self.board[y, x] = 1
        self.snake_body = [(x, y)]

    def generate_apple(self):
        # Randomly generate apple (if there isn't already body of snake)
        while True:
            indices = np.random.randint(0, high=self.x, size=2)
            y, x = indices[0], indices[1]

            if self.board[y, x] == 0:
                self.board[y, x] = 2
                self.apple_pos = (x, y)
                break
        
    def collision_with_self(self):
        # if count of eaten apples don't equel size of snake body -> collision with it self; game over
        if self.eaten_apples+1 != len(set(self.snake_body)):
            self.done = True
            self.reward = -100
    
    def collision_with_boundaries(self):
        # if snake go beyond board; game over
        if (self.snake_body[-1][0] < 0 or self.snake_body[-1][0] >= self.x) == True \
            or (self.snake_body[-1][1] < 0 or self.snake_body[-1][1] >= self.x) == True:
            self.done = True
            self.reward = -100

    def collision_with_apple(self):
        # Add another body to snake and generate another apple
        if self.snake_body[-1] == self.apple_pos:
            self.eaten_apples += 1
            self.reward = 100
            self.steps = 0

            self.check_for_end()
            if self.done == False: 
                self.generate_apple()
        else:
            self.pop_snake_tail()
        self.get_tail_dir()

    def check_steps(self):
        # If taken (row*2-row) or more steps until apple is eaten -> game over
        if self.steps > (self.x*2-self.x):
            self.done = True
    
    def check_for_end(self):
        # return True if snake filled whole board else False
        if np.all(self.board.all(1)):
            self.done = True
            self.game_info = {"finished"}
    
    def restart_game(self):
        # set up parameters
        self.eaten_apples = 1           # track of eaten apple
        self.apple_pos = None           # position of current apple
        self.head_dir = None            # direction of head of snake
        self.steps = 0                  # every action untill apple is eaten
        self.done = False               # if env/game is finished
        self.finished_game = {"not finished"} # info about state of game

        # Generate game grid
        self.generate_grid()
        self.generate_snake()
        self.generate_apple()

    def update_board(self):
        # refresh board; write on board snake and apple
        self.generate_grid()
        for body in self.snake_body:
            self.board[body[1], body[0]] = 1
        
        self.board[self.apple_pos[1], self.apple_pos[0]] = 2

    def get_state(self):
        self.Sensors.update_sensor_board(self.board, self.apple_pos, self.snake_body)
        distance_sensor = self.Sensors.distance_to_wall()
        apple_sensor = self.Sensors.see_apple()
        snake_body_sensor = self.Sensors.see_it_self()
        head_dir_sensor = self.Sensors.get_head_direction(self.head_dir)
        if len(self.snake_body) > 1: tail_dir_sensor = self.Sensors.get_tail_direction(self.tail_dir)
        else: tail_dir_sensor = np.array([0, 0, 0, 0])

        return np.concatenate((distance_sensor, apple_sensor, snake_body_sensor, head_dir_sensor, tail_dir_sensor), axis=0)
    
    def select_random_action(self):
        list_of_action = np.array([0, 1, 2, 3])
        return np.random.choice(list_of_action, size=1)[0]
    
    def get_head_dir(self, direction):
        # snake can kill himself by going opposite direction
        if direction == 0:      # up
            self.head_dir = (0, -1)
        elif direction == 1:    # right
            self.head_dir = (1, 0)
        elif direction == 2:    # down
            self.head_dir = (0, 1)
        elif direction == 3:    # left
            self.head_dir = (-1, 0)
    
    def get_tail_dir(self):
        if len(self.snake_body) > 1:
            self.tail_dir = tuple((np.array([self.snake_body[1][0], self.snake_body[1][1]] \
                 - np.array([self.snake_body[0][0], self.snake_body[0][1]]))).reshape(1, -1)[0])
    
    def pop_snake_tail(self):
        if self.eaten_apples == len(self.snake_body)-1:
            self.snake_body.pop(0)
            self.reward = -1

    def snake_move(self, direction):
        """
        Algorithm: add new part before head in corresponding direction and delete tail
        """
        self.get_head_dir(direction)
        head = self.snake_body[-1]
        self.snake_body.append((head[0]+self.head_dir[0], head[1]+self.head_dir[1]))
        # deletng tail is handled in collision_with_apple() with getting tail direction

        self.steps += 1
    
    def action(self, act):
        if self.done == True: self.restart_game()
        self.snake_move(act)
        self.collision_with_self()
        self.collision_with_boundaries()
        self.collision_with_apple()
        self.check_for_end()
        if self.done == False: self.update_board()
        return self.get_state(), self.reward, self.done, self.finished_game