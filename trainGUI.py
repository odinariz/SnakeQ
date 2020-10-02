import pygame
import numpy as np

import parameters as par
import environment
import snake_sensors
import DQNAgent

# Colors
SNAKE_C = par.SNAKE_C
APPLE_C = par.APPLE_C
BG = par.BG
APP_BG = par.APP_BG
GRID_BG = par.GRID_BG
BLACK = par.BLACK
WHITE = par.WHITE
GREY = par.GREY
GREY2 = par.GREY2
GREY3 = par.GREY3
GREY4 = par.GREY4
RED = par.RED
GREEN = par.GREEN
ORANGE = par.ORANGE
BLUE = par.BLUE
BLUE2 = par.BLUE2

class SetUp:
    def __init__(self):
        super().__init__()
        self.height = par.train_height
        self.n_row = par.row
        self.pixel = self.height // self.n_row


class DrawWindow(SetUp):
    def __init__(self):
        super().__init__()
        self.screen = pygame.display.set_mode((self.height+self.height, self.height+self.height))
        self.n_envs = 4
        self.env_height = self.height // self.n_envs

    def draw_boards(self, board_list):
        rects = []
        # draw background for game section
        rects.append(pygame.draw.rect(self.screen, BG, (0, 0, self.height+self.height, self.height+self.height)))

        i, j = 0, 0
        for index, board in enumerate(board_list):
            if index == 2:
                i = 0
                j += self.height

            # draw board
            for y, row in enumerate(board):
                for x, value in enumerate(row):
                    if value == 1:      # snake
                        rects.append(pygame.draw.rect(self.screen, SNAKE_C, (i+x*self.pixel, j+y*self.pixel, self.pixel-5, self.pixel-5)))
                    elif value == 2:    # apple
                        rects.append(pygame.draw.rect(self.screen, APPLE_C, (i+x*self.pixel, j+y*self.pixel, self.pixel-5, self.pixel-5)))
            
            i += self.height
            

        pygame.display.update(rects)
    
    def draw_lines(self):
        lines = []
        lines.append(pygame.draw.line(self.screen, BLACK, (0, self.height), (self.height+self.height, self.height), 5))
        lines.append(pygame.draw.line(self.screen, BLACK, (self.height, 0), (self.height, self.height+self.height), 5))

        pygame.display.update(lines)

def check_speed():
    global speed_up, time_delay, time_tick
    keys = pygame.key.get_pressed()
    for key in keys:
        if keys[pygame.K_SPACE] and speed_up == True:
            time_delay, time_tick = 120, 20
            speed_up = False
            return
        elif keys[pygame.K_SPACE] and speed_up == False:
            time_delay, time_tick = 0, 0
            speed_up = True
            return


if __name__ == "__main__":
    pygame.init()

    env1 = environment.Environment(par.row)
    env2 = environment.Environment(par.row)
    env3 = environment.Environment(par.row)
    env4 = environment.Environment(par.row)
    env_list = [env1, env2, env3, env4]
    buffer = DQNAgent.ExperienceBuffer(par.REPLAY_SIZE)
    net = DQNAgent.Neural_Network()
    dqn_agent = DQNAgent.DQN(env_list, buffer, net, load=False)
    win = DrawWindow()

    pygame.display.set_caption("SnakeQ by ludius0")
    clock = pygame.time.Clock()
    time_delay, time_tick = 120, 20
    speed_up = True

    while True:
        # Pygame events and times
        pygame.time.delay(time_delay)
        clock.tick(time_tick)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            check_speed()
        
        all_boards = []
        dqn_agent.simulate()
        for index in range(len(env_list)):
            board, state, epsilon, mean_reward, steps, generation, score, game_info = dqn_agent.api(index=index)
            score -= 3
            pygame.display.set_caption(f"SnakeQ by ludius0        Score: {score}    Generation: {generation}    Steps: {steps}    Epsilon: {epsilon}    Mean Reward: {mean_reward}")

            all_boards.append(board.tolist())

        win.draw_boards(all_boards)
        win.draw_lines()

        if game_info["won game"]:
            input()
            print("finished")
            break

    pygame.quit()