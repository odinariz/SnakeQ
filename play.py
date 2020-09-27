import numpy as np

import parameters as par
import environment
import snake_sensors
import DQNAgent



if __name__ == "__main__":

    # Create 4 Environments
    env1 = environment.Environment(par.row)
    env2 = environment.Environment(par.row)
    env3 = environment.Environment(par.row)
    env4 = environment.Environment(par.row)
    env_list = [env1, env2, env3, env4]

    buffer = DQNAgent.ExperienceBuffer(par.REPLAY_SIZE)
    net = DQNAgent.Neural_Network()
    dqn_agent = DQNAgent.DQN(env_list, buffer, net, load=False)

    print("List of directions: 0=up; 1=right; 2=down; 3=left")

    while True:
        dqn_agent.simulate(print_info=True, print_board=True)