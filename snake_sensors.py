import numpy as np

class SnakeSensors:
    def __init__(self, x, board_info):
        """
        Return array where are saved all information about snake and his sensors,
        which will get pass to agent through neural network for choosing action
        Sensors:
            1. distance to walls (4 dim)
            2. distance to body snake (8 dim)
            3. distance to apple (8 dim)
            Note: distance is converted between 0.0 to 1.0
        """
        self.dis = x    # row/x/distance
        self.target = board_info
        self.start = {"forward": 1, "backward": -1}
    
    def get_head_direction(self, head_dir):
        if head_dir == (0, -1): return np.array([1, 0, 0, 0])   # up
        elif head_dir == (1, 0): return np.array([0, 1, 0, 0])  # right
        elif head_dir == (0, 1): return np.array([0, 0, 1, 0])  # down
        elif head_dir == (-1, 0): return np.array([0, 0, 0, 1]) # left
    
    def get_tail_direction(self):
        if self.tail_dir == (0, -1): return np.array([1, 0, 0, 0])  # up
        elif self.tail_dir == (1, 0): return np.array([0, 1, 0, 0]) # right
        elif self.tail_dir == (0, 1): return np.array([0, 0, 1, 0]) # down
        elif self.tail_dir == (-1, 0): return np.array([0, 0, 0, 1])# left
    
    def check_horizontal(self, target, from_head_x, to_boundary, start, skip_first):
        # left: range(self.head_x, 0);  right: range(self.head_x, self.dis);
        for i in range(from_head_x, to_boundary, start):
            if i == from_head_x and skip_first == True: continue
            if self.board[self.head_y, i] == target:
                return 1    # True
        return 0    # False
    
    def check_vertical(self, target, from_head_y, to_boundary, start, skip_first):
        # up: range(self.head_y, 0);  down: range(self.head_y, self.dis)
        for i in range(from_head_y, to_boundary, start):
            if i == from_head_y and skip_first == True: continue
            if self.board[i, self.head_x] == target:
                return 1
        return 0
    
    def check_right_up(self, target, skip_first):
        distance = self.dis_y_up if self.dis_y_up < self.dis_x_right else self.dis_x_right
        if distance == 0: return 0
        for n, _ in enumerate(range(distance, 0, -1)):
            if n == 0 and skip_first == True: continue
            if self.board[self.head_y-n, self.head_x+n] == target: 
                return 1
        return 0
    
    def check_right_down(self, target, skip_first):
        distance = self.dis_y_down if self.dis_y_down < self.dis_x_right else self.dis_x_right
        if distance == self.dis: return 0
        for n, _ in enumerate(range(0, distance)):
            if n == 0 and skip_first == True: continue
            if self.board[self.head_y+n, self.head_x+n] == target: 
                return 1
        return 0
    
    def check_left_up(self, target, skip_first):
        distance = self.dis_y_up if self.dis_y_up < self.dis_x_left else self.dis_x_left
        if distance == 0: return 0
        for n, _ in enumerate(range(distance, 0, -1)):
            if n == 0 and skip_first == True: continue
            if self.board[self.head_y-n, self.head_x-n] == target: 
                return 1
        return 0

    def check_left_down(self, target, skip_first):
        distance = self.dis_y_down if self.dis_y_down < self.dis_x_left else self.dis_x_left
        if distance == self.dis: return 0
        for n, _ in enumerate(range(0, distance)):
            if n == 0 and skip_first == True: continue
            if self.board[self.head_y+n, self.head_x-n] == target: 
                return 1
        return 0
    
    def update_board(self, board, apple, snake):
        self.board = board
        self.apple_pos = apple
        self.snake_body = snake
        if len(self.snake_body) > 1:
            self.tail_dir = tuple((np.array([self.snake_body[1][0], self.snake_body[1][1]] - np.array([self.snake_body[0][0], self.snake_body[0][1]]))).reshape(1, -1)[0])
        self.head_x, self.head_y = self.snake_body[-1]

        # distance to boundaries in 4 dimensions
        self.dis_y_down = self.dis - self.head_y - 1    # -1 because grid is from 0 to boundary
        self.dis_y_up = self.dis - self.dis_y_down - 1
        self.dis_x_right = self.dis - self.head_x - 1
        self.dis_x_left = self.dis - self.dis_x_right - 1
    
    def all_eight_directions(self, target, skip_first=False):
        to_up = self.check_vertical(target, self.head_y, 0, self.start["backward"], skip_first)               # up
        to_right_up = self.check_right_up(target, skip_first)                                                 # right up
        to_right = self.check_horizontal(target, self.head_x, self.dis, self.start["forward"], skip_first)    # right
        to_right_down = self.check_right_down(target, skip_first)                                             # right down
        to_down = self.check_vertical(target, self.head_y, self.dis, self.start["forward"], skip_first)       # down
        to_left_down = self.check_left_down(target, skip_first)                                               # left down
        to_left = self.check_horizontal(target, self.head_x, 0, self.start["backward"], skip_first)           # left
        to_left_up = self.check_left_up(target, skip_first)                                                   # left down
        return np.array([to_up, to_right_up, to_right, to_right_down, to_down, to_left_down, to_left, to_left_up])
    
    def distance_to_wall(self):
        return np.round(np.array([self.dis_y_up, self.dis_x_right, self.dis_y_down, self.dis_x_left]) / self.dis, 1)

    def see_apple(self):
        return self.all_eight_directions(self.target["apple"])

    def see_it_self(self):
        return self.all_eight_directions(self.target["snake"], skip_first=True) # skip head