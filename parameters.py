# COLORS
SNAKE_C = (209, 204, 192)
APPLE_C = (192, 57, 43) #(255, 195, 18) #(255, 177, 66)
BG = (44, 44, 84) # background
APP_BG = (240, 240, 240)
GRID_BG = (44, 44, 84)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (113, 128, 147)
GREY2 = (189, 195, 199)
GREY3 = (87, 96, 111)
GREY4 = (53, 59, 72)
RED = (229, 57, 53)
GREEN = (32, 191, 107)
ORANGE = (250, 130, 49)
BLUE = (7, 153, 146)
BLUE2 = (30, 55, 153)

# GUI and ENV (*Numbers have to be rounded)
pixel = 100
width_height = 800
app_width = 800
app_height = 600
row = 10
fps = 200

# Widgets
MainWidgetSize = (0, 0, 1000, 1000)
EnvWidgetSize = (0, 0, 600, 600)

# Neural Network and Q-learning
GAMMA = 0.99  # discount
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 0.02 #1e-4
SYNC_TARGET_LOOPS = 1000
REPLAY_START_SIZE = 10000

# epsilon -> chance for random action
EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02