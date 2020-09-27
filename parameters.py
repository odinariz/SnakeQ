# COLORS
SNAKE_C = (209, 204, 192)
APPLE_C = (255, 177, 66)
BG = (44, 44, 84) # background
GRID_BG = (44, 44, 84)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# GUI and ENV (*Numbers have to be rounded)
pixel = 100
width_height = 300
row = width_height // pixel
fps = 200

# Widgets
MainWidgetSize = (0, 0, 1000, 1000)
EnvWidgetSize = (0, 0, 600, 600)

# Neural Network and Q-learning
GAMMA = 0.99  # discount
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_LOOPS = 1000
REPLAY_START_SIZE = 10000

# epsilon -> chance for random action
EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02