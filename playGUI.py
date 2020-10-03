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
        self.width = par.app_width
        self.width_plus = 200
        self.height = par.app_height
        self.sensor_space = self.width - self.height
        self.sensor_n_row = self.sensor_space // 3
        self.n_row = par.row
        self.pixel = self.height // self.n_row

class DrawSensors(SetUp):
    def __init__(self):
        super().__init__()
        self.screen = pygame.display.set_mode((par.app_width+self.width_plus, par.app_height))
    
    def reshuffle_state(self, state):
        # reshaffle for 3x3 grid loop logic
        return [state[7], state[0], state[1], state[6], 0, state[2], state[5], state[4], state[3]]

    def draw_distance(self, state):
        state = [x*255 for x in state]
        state_ = [state[0], state[3], state[1], state[2]]
        ignore = ((0, 0), (2, 0), (1, 1), (0, 2), (2, 2))
        distance_list = []

        index = 0
        for y in range(0, 3):
            for x in range(0, 3):
                if not (y, x) in ignore:
                    distance_list.append(pygame.draw.rect(self.screen, (state_[index], state_[index], state_[index]), \
                        ((self.height+self.sensor_n_row*x+5), self.sensor_n_row*y, self.sensor_n_row-5, self.sensor_n_row-5)))
                    index += 1
        pygame.display.update(distance_list)

    def draw_apple(self, state):
        state_ = self.reshuffle_state(state)
        see_apple_list = []

        index = 0
        for y in range(0, 3):
            for x in range(0, 3):
                if (y, x) != (1, 1):
                    if state_[index] == 1:  # Draw True
                        see_apple_list.append(pygame.draw.rect(self.screen, APPLE_C, \
                            ((self.height+self.sensor_n_row*x+5), self.sensor_n_row*3+self.sensor_n_row*y, self.sensor_n_row-5, self.sensor_n_row-5)))
                    else:
                        see_apple_list.append(pygame.draw.rect(self.screen, GREY2, \
                            ((self.height+self.sensor_n_row*x+5), self.sensor_n_row*3+self.sensor_n_row*y, self.sensor_n_row-5, self.sensor_n_row-5)))
                index += 1
        pygame.display.update(see_apple_list)

    def draw_see_self(self, state):
        state_ = self.reshuffle_state(state)
        see_self_list = []

        index = 0
        for y in range(0, 3):
            for x in range(0, 3):
                if (y, x) != (1, 1):
                    if state_[index] == 1:  # Draw True
                        see_self_list.append(pygame.draw.rect(self.screen, GREY3, \
                            ((self.height+self.sensor_n_row*x+5), self.sensor_n_row*6+self.sensor_n_row*y, self.sensor_n_row-5, self.sensor_n_row-5)))
                    else:
                        see_self_list.append(pygame.draw.rect(self.screen, GREY, \
                            ((self.height+self.sensor_n_row*x+5), self.sensor_n_row*6+self.sensor_n_row*y, self.sensor_n_row-5, self.sensor_n_row-5)))
                index += 1
        pygame.display.update(see_self_list)



class DrawWindow(DrawSensors):
    def __init__(self):
        super().__init__()
        self.text_strings = ["Score:", "Episode:", "Generation:", "Steps:", "Epsilon:", "Mean Reward:"]
        self.text_range = 30
    
    def draw_sensor_bg(self):
        pygame.display.update(pygame.draw.rect(self.screen, APP_BG, (self.height+1, 0, self.width+self.width_plus, self.height)))

    def draw_board(self, board):
        rects = []
        # draw background for game section
        rects.append(pygame.draw.rect(self.screen, BG, (0, 0, self.height+1, self.height+1)))

        # draw board
        for y, row in enumerate(board):
            for x, value in enumerate(row):
                if value == 1:      # snake
                    rects.append(pygame.draw.rect(self.screen, SNAKE_C, (x*self.pixel, y*self.pixel, self.pixel-5, self.pixel-5)))
                elif value == 2:    # apple
                    rects.append(pygame.draw.rect(self.screen, APPLE_C, (x*self.pixel, y*self.pixel, self.pixel-5, self.pixel-5)))
        pygame.display.update(rects)

    def draw_sensors(self, state):
        self.draw_distance(state[0:4])
        self.draw_apple(state[4:12])
        self.draw_see_self(state[12:20])
    
    def draw_text(self, text_list):
        myfont = pygame.font.SysFont("freesansbold.ttf", 32)
        for index in range(len(text_list)):
            textsurface = myfont.render(f"{self.text_strings[index]}", False, BLACK)
            textRect = textsurface.get_rect()
            textRect.center = (self.width-(self.width_plus//2)+self.width_plus, self.text_range*index*3+self.text_range*2) 
            self.screen.blit(textsurface, textRect)

            if text_list[index] != None:
                textsurface2 = myfont.render(f"{round(text_list[index], 2)}", False, BLACK)
                textRect2 = textsurface2.get_rect()
                textRect2.center = (self.width-(self.width_plus//2)+self.width_plus, self.text_range*index*3+self.text_range*2) 
                self.screen.blit(textsurface2, (textRect2.center[0]-10, textRect2.center[1]+self.text_range-10))
        pygame.display.flip()

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
    pygame.font.init()

    env = environment.Environment(par.row)
    buffer = DQNAgent.ExperienceBuffer(par.REPLAY_SIZE)
    net = DQNAgent.Neural_Network()
    dqn_agent = DQNAgent.DQN(env, buffer, net, load=True)
    win = DrawWindow()

    pygame.display.set_caption("SnakeQ by ludius0")
    clock = pygame.time.Clock()
    time_delay, time_tick = 120, 20
    speed_up = True

    while True:
        # Pygame events and time
        pygame.time.delay(time_delay)
        clock.tick(time_tick)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            check_speed()

        dqn_agent.simulate()
        board, state, epsilon, mean_reward, steps, generation, score, episode, game_info = dqn_agent.api()
        board, state = board.tolist(), state.tolist()
        score -= 3

        pygame.display.set_caption(f"SnakeQ by ludius0        Score: {score}    Episode: {episode}   \
            Generation: {generation}    Steps: {steps}    Epsilon: {epsilon}    Mean Reward: {mean_reward}")

        win.draw_sensor_bg()
        win.draw_sensors(state)
        win.draw_board(board)
        win.draw_text([score, episode, generation, steps, epsilon, mean_reward])

        if game_info["won game"]:
            print("finished")
            input()
            break

    pygame.quit()