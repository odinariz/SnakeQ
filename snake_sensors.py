import numpy as np

class SnakeSensors:
    def __init__(self, x, board_info, moves):
        """
        Return array where are saved all information about snake and his sensors,
        which will get pass to agent through neural network for choosing action
        Sensors:
            1. distance to walls (4 dim)
            2. distance to apple (8 dim)
            3. distance to body snake (8 dim)
            4. snake head direction
            5. snake tail direction
            Note: distance is converted between 0.0 to 1.0
        """
        self.dis = x    # row/x/distance
        self.target = board_info
        self.moves = moves
    
    def update_sensor_board(self, board, apple, snake):
        self.board = board
        self.apple_pos = apple
        self.head_y, self.head_x = snake[-1]

        # distance to boundaries in 4 dimensions
        self.dis_y_down = self.dis - self.head_y - 1    # -1 because grid is from 0 to boundary
        self.dis_y_up = self.dis - self.dis_y_down - 1
        self.dis_x_right = self.dis - self.head_x - 1
        self.dis_x_left = self.dis - self.dis_x_right - 1
    
    def check_up(self, target):
        if self.head_y == 0: return 0
        for i in range(self.head_y-1, -1, -1):
            if self.board[i, self.head_x] == target:
                return 1
        return 0
    
    def check_down(self, target):
        if self.head_y == self.dis-1: return 0
        for i in range(self.head_y+1, self.dis):
            if self.board[i, self.head_x] == target:
                return 1
        return 0
    
    def check_right(self, target):
        if self.head_x == self.dis-1: return 0
        for i in range(self.head_x+1, self.dis):
            if self.board[self.head_y, i] == target:
                return 1
        return 0
    
    def check_left(self, target):
        if self.head_x == 0: return 0
        for i in range(self.head_x-1, -1, -1):
            if self.board[self.head_y, i] == target:
                return 1
        return 0
    
    def check_right_up(self, target):
        # choose shorter distance
        distance = self.dis_y_up if self.dis_y_up < self.dis_x_right else self.dis_x_right
        if distance == 0: return 0
        for n in range(1, distance+1):
            if self.board[self.head_y-n, self.head_x+n] == target:
                return 1
        return 0
    
    def check_right_down(self, target):
        distance = self.dis_y_down if self.dis_y_down < self.dis_x_right else self.dis_x_right
        if distance == self.dis: return 0
        for n in range(1, distance+1):
            if self.board[self.head_y+n, self.head_x+n] == target: 
                return 1
        return 0
    
    def check_left_up(self, target):
        distance = self.dis_y_up if self.dis_y_up < self.dis_x_left else self.dis_x_left
        if distance == 0: return 0
        for n in range(1, distance+1):
            if self.board[self.head_y-n, self.head_x-n] == target: 
                return 1
        return 0

    def check_left_down(self, target):
        distance = self.dis_y_down if self.dis_y_down < self.dis_x_left else self.dis_x_left
        if distance == self.dis: return 0
        for n in range(1, distance+1):
            if self.board[self.head_y+n, self.head_x-n] == target: 
                return 1
        return 0
    
    def all_eight_directions(self, target):
        to_up = self.check_up(target)                       # up
        to_right_up = self.check_right_up(target)           # right up
        to_right = self.check_right(target)                 # right
        to_right_down = self.check_right_down(target)       # right down
        to_down = self.check_down(target)                   # down
        to_left_down = self.check_left_down(target)         # left down
        to_left = self.check_left(target)                   # left
        to_left_up = self.check_left_up(target)             # left up
        return np.array([to_up, to_right_up, to_right, to_right_down, to_down, to_left_down, to_left, to_left_up])
    
    def distance_to_wall(self):
        return np.round(np.array([self.dis_y_up, self.dis_x_right, self.dis_y_down, self.dis_x_left]) / self.dis, 1)

    def see_apple(self):
        return self.all_eight_directions(self.target["apple"])

    def see_it_self(self):
        return self.all_eight_directions(self.target["snake"]) # skip snake head
    
    def get_head_direction(self, head_dir):
        if np.array_equal(head_dir, self.moves["up"]): return np.array([1, 0, 0, 0])   # up
        elif np.array_equal(head_dir, self.moves["right"]): return np.array([0, 1, 0, 0])  # right
        elif np.array_equal(head_dir, self.moves["down"]): return np.array([0, 0, 1, 0])  # down
        elif np.array_equal(head_dir, self.moves["left"]): return np.array([0, 0, 0, 1]) # left
    
    def get_tail_direction(self, tail_dir):
        if np.array_equal(tail_dir, self.moves["up"]): return np.array([1, 0, 0, 0])  # up
        elif np.array_equal(tail_dir, self.moves["right"]): return np.array([0, 1, 0, 0]) # right
        elif np.array_equal(tail_dir, self.moves["down"]): return np.array([0, 0, 1, 0]) # down
        elif np.array_equal(tail_dir, self.moves["left"]): return np.array([0, 0, 0, 1])# left