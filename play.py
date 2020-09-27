import numpy as np

import parameters as p
import environment
import snake_sensors
import DQNAgent



if __name__ == "__main__":

    net = DQNAgent.Neural_Network()
    agent = DQNAgent.QAgent(net)
    env = environment.Environment(p.row)

    print("List of directions: 0=up; 1=right; 2=down; 3=left")

    # build testing console game
    while True:
        print(env.board)

        direction = int(input("Direction: "))
        state, reward, done, info = env.action(direction)
        print("states", state)
        print("reward", reward)

        if env.done == True:
            print("Game Over!")
            break