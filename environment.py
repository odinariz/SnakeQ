import numpy as np
import random

import snake_sensors

class Environment:
    def __init__(self, row):
        # parameters
        self.board_info = {"empty": 0, "snake": 1, "apple": 2}
        # (y, x)
        self.moves = {"up": np.array([-1, 0]), "right": np.array([0, 1]), "down": np.array([1, 0]), "left": np.array([0, -1])}
        self.reward_dict = {"hit self": -100, "hit boundary": -100, "eat apple": 30, "step": -1, "win game": 100}
        # Snake sensors for returning state (all logic)
        self.Sensors = snake_sensors.SnakeSensors(row, self.board_info, self.moves)

        # Generate components
        self.x = row
        self.restart_env()
    
    def generate_grid(self):
        # generate board (grid) of zeros (always square)
        self.board = np.zeros((self.x, self.x))
    
    def generate_snake(self):
        # Randomly choose spot to generate snake
        indices = np.random.randint(0, high=self.x, size=2)
        y, x = indices[0], indices[1]

        self.board[y, x] = 1
        self.snake_body = np.array([[y, x]])

    def generate_apple(self):
        # Randomly generate apple (if there isn't already body of snake)
        while True:
            indices = np.random.randint(0, high=self.x, size=2)
            y, x = indices[0], indices[1]

            if self.board[y, x] == 0:
                self.board[y, x] = 2
                self.apple_pos = np.array([y, x])
                break
        
    def collision_with_self(self):
        # if count of eaten apples don't equel size of snake body -> collision with it self; game over
        if self.eaten_apples+1 != len(self.snake_body.astype(set)):
            self.done = True
            self.reward = self.reward_dict["hit self"]
            print("Collision with self")
    
    def collision_with_boundaries(self):
        # if snake go beyond board; game over
        if (self.snake_body[-1, 0] < 0 or self.snake_body[-1, 0] >= self.x) == True \
            or (self.snake_body[-1, 1] < 0 or self.snake_body[-1, 1] >= self.x) == True:
            self.done = True
            self.reward = self.reward_dict["hit boundary"]
            print("Collision with boundaries")

    def collision_with_apple(self):
        # Add another body to snake and generate another apple
        if np.array_equal(self.snake_body[-1], self.apple_pos):
            self.eaten_apples += 1
            self.reward = self.reward_dict["eat apple"]
            self.steps = 0

            self.check_for_end()
            if self.done == False: 
                self.generate_apple()
            print("Collision with apple")
        else:
            self.pop_snake_tail()
        self.get_tail_dir()

    def check_steps(self):
        # If taken (row*4) or more steps until apple is eaten -> game over
        if self.steps > self.x*4:
            print("a lot of steps")
            print(self.eaten_apples)
            self.done = True
    
    def check_for_end(self):
        # return True if snake filled whole board else False
        if np.all(self.board.all(1)):
            self.done = True
            self.game_info = {"won game": True}
    
    def restart_env(self):
        # set up parameters
        self.eaten_apples = 1           # track of eaten apple
        self.apple_pos = None           # position of current apple
        self.none = None
        self.head_dir = self.none       # direction of head of snake
        self.steps = 0                  # every action untill apple is eaten
        self.reward = 0                 # reward based on every action and its consequences
        self.done = False               # if env/game is finished
        self.game_info = {"won game": False} # info about state of game

        # Generate game grid
        self.generate_grid()
        self.generate_snake()
        self.generate_apple()
        
        return self.get_state(), self.reward, self.done, self.game_info

    def update_board(self):
        # refresh board; write on board snake and apple
        self.generate_grid()
        for body in self.snake_body:
            self.board[body[0], body[1]] = 1
        
        self.board[self.apple_pos[0], self.apple_pos[1]] = 2

    def get_state(self):
        self.Sensors.update_sensor_board(self.board, self.apple_pos, self.snake_body)
        distance_sensor = self.Sensors.distance_to_wall()
        apple_sensor = self.Sensors.see_apple()
        snake_body_sensor = self.Sensors.see_it_self()
        if self.head_dir is not self.none: head_dir_sensor = self.Sensors.get_head_direction(self.head_dir)
        else: head_dir_sensor = np.array([0, 0, 0, 0])

        if len(self.snake_body) > 1: tail_dir_sensor = self.Sensors.get_tail_direction(self.tail_dir)
        else: tail_dir_sensor = np.array([0, 0, 0, 0])

        return np.concatenate((distance_sensor, apple_sensor, snake_body_sensor, head_dir_sensor, tail_dir_sensor), axis=0)
    
    def select_random_action(self):
        list_of_action = np.array([0, 1, 2, 3])
        return np.random.choice(list_of_action, size=1)[0]
    
    def get_head_dir(self, direction):
        # snake can kill himself by going opposite direction
        if direction == 0:      self.head_dir = self.moves["up"]
        elif direction == 1:    self.head_dir = self.moves["right"]
        elif direction == 2:    self.head_dir = self.moves["down"]
        elif direction == 3:    self.head_dir = self.moves["left"]
    
    def get_tail_dir(self):
        if len(self.snake_body) > 1:
            self.tail_dir = tuple((np.array([self.snake_body[1, 0], self.snake_body[1, 1]] \
                 - np.array([self.snake_body[0, 0], self.snake_body[0, 1]]))).reshape(1, -1)[0])
    
    def pop_snake_tail(self):
        if self.eaten_apples == len(self.snake_body)-1 and self.done == False:
            self.snake_body = np.delete(self.snake_body, 0, 0)
            self.reward = self.reward_dict["step"]

    def snake_move(self, direction):
        """
        Algorithm: add new part before head in corresponding direction and delete tail
        """
        self.get_head_dir(direction)
        head = self.snake_body[-1]
        new_head = np.array([head[0]+self.head_dir[0], head[1]+self.head_dir[1]])
        self.snake_body = np.vstack((self.snake_body, new_head))
        # deletng tail is handled in collision_with_apple() with getting tail direction and reward for nothing happening

        self.steps += 1
    
    def action(self, act):
        if self.done == True: self.restart_env()
        self.snake_move(act)
        self.collision_with_self()
        self.collision_with_boundaries()
        self.collision_with_apple()
        self.check_steps()
        self.check_for_end()
        if self.done == False: self.update_board()
        else: self.restart_env()
        return self.get_state(), self.reward, self.done, self.game_info