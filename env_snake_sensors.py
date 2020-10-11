import numpy as np

class SnakeSensors:
    def __init__(self, row, moves):
        self.row = row
        self.dis = self.row-1
        self.moves = moves

        self.next_to_head_dir = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    def update_sensor_board(self, board, snake):
        self.board = board
        self.head_y, self.head_x = snake

        # distance to boundaries in 4 dimensions
        self.dis_y_down = self.dis - self.head_y
        self.dis_y_up = self.dis - self.dis_y_down
        self.dis_x_right = self.dis - self.head_x
        self.dis_x_left = self.dis - self.dis_x_right

    def check_up(self, target):
        if self.head_y == 0: return 0
        for i in range(self.head_y-1, -1, -1):
            if self.board[i, self.head_x] == target:
                return i
        return 0
    
    def check_down(self, target):
        if self.head_y == self.dis: return 0
        for i in range(self.head_y+1, self.row):
            if self.board[i, self.head_x] == target:
                return i
        return 0
    
    def check_right(self, target):
        if self.head_x == self.dis: return 0
        for i in range(self.head_x+1, self.row):
            if self.board[self.head_y, i] == target:
                return i
        return 0
    
    def check_left(self, target):
        if self.head_x == 0: return 0
        for i in range(self.head_x-1, -1, -1):
            if self.board[self.head_y, i] == target:
                return i
        return 0
    
    def check_right_up(self, target):
        # choose shorter distance
        distance = self.dis_y_up if self.dis_y_up < self.dis_x_right else self.dis_x_right
        if distance == 0: return 0
        for n in range(1, distance+1):
            if self.board[self.head_y-n, self.head_x+n] == target:
                return n
        return 0
    
    def check_right_down(self, target):
        distance = self.dis_y_down if self.dis_y_down < self.dis_x_right else self.dis_x_right
        if distance == self.row: return 0
        for n in range(1, distance+1):
            if self.board[self.head_y+n, self.head_x+n] == target: 
                return n
        return 0
    
    def check_left_up(self, target):
        distance = self.dis_y_up if self.dis_y_up < self.dis_x_left else self.dis_x_left
        if distance == 0: return 0
        for n in range(1, distance+1):
            if self.board[self.head_y-n, self.head_x-n] == target: 
                return n
        return 0

    def check_left_down(self, target):
        distance = self.dis_y_down if self.dis_y_down < self.dis_x_left else self.dis_x_left
        if distance == self.row: return 0
        for n in range(1, distance+1):
            if self.board[self.head_y+n, self.head_x-n] == target: 
                return n
        return 0
    
    def all_eight_directions(self, target):
        to_up = self.check_up(target)                       # up
        to_right_up = self.check_right_up(target)           # up right
        to_right = self.check_right(target)                 # right
        to_right_down = self.check_right_down(target)       # right down
        to_down = self.check_down(target)                   # down
        to_left_down = self.check_left_down(target)         # down left
        to_left = self.check_left(target)                   # left
        to_left_up = self.check_left_up(target)             # left up
        return np.array([to_up, to_right_up, to_right, to_right_down, to_down, to_left_down, to_left, to_left_up])

    def next_to_head(self, target):
        array = np.array([])
        for y, x in self.next_to_head_dir:
          if self.head_y+y > -1 and self.head_y+y < self.dis \
            and self.head_x+x > -1 and self.head_x+x < self.dis:
            if self.board[self.head_y+y, self.head_x+x] == target:
              array = np.append(array, np.array([1]))
            else:   array = np.append(array, np.array([0]))
          else:     array = np.append(array, np.array([0]))
        return array
    
    def distance_to_walls(self):
        return np.round(np.array([self.dis_y_up, self.dis_x_right, self.dis_y_down, self.dis_x_left]) / self.dis, 1)

    def get_head_direction(self, head_dir):
        if type(head_dir) == type(None):
          return np.array([0, 0, 0, 0])
        else:
          if np.array_equal(head_dir, self.moves["up"]): return np.array([1, 0, 0, 0])        # up
          elif np.array_equal(head_dir, self.moves["right"]): return np.array([0, 1, 0, 0])   # right
          elif np.array_equal(head_dir, self.moves["down"]): return np.array([0, 0, 1, 0])    # down
          elif np.array_equal(head_dir, self.moves["left"]): return np.array([0, 0, 0, 1])    # left

    def get_tail_direction(self, snake):
        if len(snake)> 1:
            tail_dir = tuple((np.array([snake[1, 0], snake[1, 1]] \
                 - np.array([snake[0, 0], snake[0, 1]]))).reshape(1, -1)[0])
            if np.array_equal(tail_dir, self.moves["up"]): return np.array([1, 0, 0, 0])      # up
            elif np.array_equal(tail_dir, self.moves["right"]): return np.array([0, 1, 0, 0]) # right
            elif np.array_equal(tail_dir, self.moves["down"]): return np.array([0, 0, 1, 0])  # down
            elif np.array_equal(tail_dir, self.moves["left"]): return np.array([0, 0, 0, 1])  # left
        else:
            return np.array([0, 0, 0, 0])