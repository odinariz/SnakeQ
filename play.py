import numpy as np

import parameters as p
import environment
import DQNAgent



if __name__ == "__main__":

    net = DQNAgent.Neural_Network()
    agent = DQNAgent.QAgent(net)
    env = environment.Environment(p.row)

    print("List of directions: 0=up; 1=down; 2=left; 3=right")

    # build testing console game
    while True:
        print(env.board)

        direction = int(input("Direction: "))
        state, reward, done, info = env.action(direction)

        if env.done == True:
            print("Game Over!")
            break