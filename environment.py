import numpy as np

from env_snake_sensors import *

class Environment():
############################### Initalize parameters ###############################
    def __init__(self, n_row):
        # row * row = board
        self.row = n_row
        # map of board
        self.blocks = {"empty": 0, "snake": 1, "apple": 2}
        # (y, x)
        self.moves_dir = {"up": np.array([-1, 0]), "right": np.array([0, 1]), \
                      "down": np.array([1, 0]), "left": np.array([0, -1])}
        # List of all rewards
        self.reward_dict = {"hit self": -100, "hit boundary": -100, "eat apple": 10, \
                            "step": -1, "a lot of steps": -100, "win game": 1000}
        # Number of possible actions
        self.action_space = 4
        # Set up sensors for getting state
        self.Sensors = SnakeSensors(self.row, self.moves_dir)
        # Prepare game
        self.reset()

    def second_init(self):
        self.done = False           # If game is over (death)
        self.direction = None       # Direction of head (for computing state)
        self.steps = 0              # count of steps until it reache apple
        self.eaten_apples = 0       # count of eaten apples
        self.info = "Unfinished"    # If player win the game

############################### GENERATING ###############################
    def generate_grid(self):
        # generate board (grid) of zeros (always square)
        self.board = np.zeros((self.row, self.row))
    
    def generate_snake(self):
        # Randomly choose spot to generate snake
        indices = np.random.randint(0, high=self.row, size=2)
        y, x = indices[0], indices[1]

        self.board[y, x] = self.blocks["snake"]
        self.snake_body = np.array([[y, x]])
        self.beginning_lenght = 1

    def generate_apple(self):
        # Randomly generate apple (if there isn't already body of snake)
        while True:
            indices = np.random.randint(0, high=self.row, size=2)
            y, x = indices[0], indices[1]

            if self.board[y, x] == self.blocks["empty"]:
                self.board[y, x] = self.blocks["apple"]
                self.apple_pos = np.array([y, x])
                break

############################### CHECK LOGIC ###############################
    def check_n_steps(self):
        # If count of steps is bigger than treshold; game over
        if self.steps > (self.row**2/2):
            self.done = True
            self.reward = self.reward_dict["a lot of steps"]

    def check_hit_self(self):
        # Check if set of body isn't long as it had eaten apples
        self.body = set([(i[0], i[1]) for i in self.snake_body.tolist()])
        self.len_body = len(self.body)
        if len(self.body) != self.eaten_apples+self.beginning_lenght:
            self.done = True
            self.reward = self.reward_dict["hit self"]

    def check_boundaries(self, new_head):
        # Check if (y, x) go beyond boundary
        y, x = new_head
        if y < 0 or x < 0 or y > self.row-1 or x > self.row-1:
            self.done = True
            self.reward = self.reward_dict["hit boundary"]
    
    def check_end_of_game(self):
        # If whole board is filled with snake; player won
        if np.all(self.board.all(self.blocks["snake"])):
            self.done = True
            self.reward = self.reward_dict["win game"]
            self.info = "Finished"
    
    def check_eaten_apple(self, head):
        # If head is on position of apple; restart steps and update other components...
        if np.array_equal(head, self.apple_pos):
            self.steps = 0
            self.eaten_apples += 1
            self.generate_apple()
            self.reward = self.reward_dict["eat apple"] + len(self.snake_body)**2
            return True
        return False

    def snake_algorithm(self, new_head):
        # Set new head of snake before current head in corresponding direction
        self.snake_body = np.vstack((self.snake_body, new_head))

        # if eaten apple == False; tail is deleted
        if not self.check_eaten_apple(self.snake_body[-1]):
            self.snake_body = np.delete(self.snake_body, 0, 0)
    
    def move(self, action):
        # handling whole logic
        if action == 0:     direction = self.moves_dir["up"]
        elif action == 1:   direction = self.moves_dir["right"]
        elif action == 2:   direction = self.moves_dir["down"]
        elif action == 3:   direction = self.moves_dir["left"]
        self.direction = direction
        head_pos = self.snake_body[-1]
        new_head_pos = (head_pos[0]+direction[0], head_pos[1]+direction[1])

        self.check_n_steps()
        self.check_hit_self()
        self.check_boundaries(new_head_pos)
        if not self.done:
            self.snake_algorithm(new_head_pos)

    def compute_state(self):
        # Compute state of snake sensors from SnakeSensors; get passed to Agent
        self.Sensors.update_sensor_board(self.board, self.snake_body[-1])
        #next_to_head = self.Sensors.next_to_head(self.blocks["empty"])
        distance = self.Sensors.distance_to_walls()
        see_apple = self.Sensors.all_eight_directions(self.blocks["apple"])
        see_self = self.Sensors.all_eight_directions(self.blocks["snake"])
        head_dir = self.Sensors.get_head_direction(self.direction)
        tail_dir = self.Sensors.get_tail_direction(self.snake_body)
        self.state = np.concatenate((distance, see_apple, see_self, head_dir, tail_dir), axis=0)

############################### PERFORM FUNCTIONS FOR ENV ###############################
    def sample_action(self):
        # return random action
        return np.random.choice(np.array([0, 1, 2, 3]), size=1)[0]
    
    def refresh_board(self):
        # refresh board; write on board snake and apple
        self.generate_grid()
        for body in self.snake_body:
            self.board[body[0], body[1]] = self.blocks["snake"]
        self.board[self.apple_pos[0], self.apple_pos[1]] = self.blocks["apple"]

    def reset(self):
        # Reset/set up game parameters
        self.second_init()

        # Generate and refresh board
        self.generate_grid()
        self.generate_snake()
        self.generate_apple()
        self.compute_state()

        return self.state

    def step(self, action):
        # Perform action, whole back up logic and return results of action
        self.reward = self.reward_dict["step"]
        self.steps += 1
        if self.done:
            self.reset()

        self.move(action)
        self.compute_state()
        self.refresh_board()

        return self.state, self.reward, self.done, self.info