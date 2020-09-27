import numpy as np

import parameters as par
import environment
import snake_sensors
import DQNAgent



if __name__ == "__main__":

    env = environment.Environment(par.row)
    buffer = DQNAgent.ExperienceBuffer(par.REPLAY_SIZE)
    net = DQNAgent.Neural_Network()
    dqn_agent = DQNAgent.DQN(env, buffer, net, load=False)

    print("List of directions: 0=up; 1=right; 2=down; 3=left")

    dqn_agent.simulate(print_board=True)