import collections
import numpy as np

import parameters as par

import torch
import torch.nn as nn

class Neural_Network(nn.Module):
    def __init__(self, input_size=28, lr=par.LEARNING_RATE):
        super().__init__()
        """
        Input to NN:
            [distance to wall, see apple, see it self, head direction, tail direction] -> 28 elements
        output of NN:
            [0: up    1: right    2: down    3: left] -> 4 elements
        """
        # Neural Network
        self.layer_neurons = {"Input Size": 28, "First Hidden Layer": 20, "Second Hidden Layer": 12, "Output Size": 4}
        self.activation_function = {"ReLU": nn.ReLU(), "Sigmoid": nn.Sigmoid()}
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(),
            nn.Linear(20, 12),
            nn.ReLU(),
            nn.Linear(12, 4),
            nn.Sigmoid()
        )

        self.loss_f = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, input_tensor):
        return self.model(input_tensor)

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)

class Agent:
    def __init__(self, env, buffer):
        self.env_list = env if type(env) is type([]) else [env] 
        self.buffer = buffer
        self.Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])
        for env_ in self.env_list:
            self._reset(env_)

    def _reset(self, env):
        self.state = env.restart_env()
        self.total_reward = 0.0

    def play_step(self, net, env, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            act = env.select_random_action()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.from_numpy(state_a).to(device, dtype=torch.float32)
            q_vals_v = net.forward(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            act = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = env.action(act)
        self.total_reward += reward

        exp = self.Experience(self.state, act, reward, is_done, new_state)
        self.buffer.append(exp)
        self.state = new_state

        if is_done:
            done_reward = self.total_reward
            self._reset(env)
        return done_reward

class DQN(Agent):
    def __init__(self, env, buffer, net, load):
        super().__init__(env, buffer)
        self.device = self.select_device()
        # Netwroks
        self.net = Neural_Network().to(self.device)
        self.target_net = Neural_Network().to(self.device)

        # parameters
        self.total_rewards = []
        self.best_mean_reward = None
        self.mean_reward = None
        self.index = 0
        self.epsilon = par.EPSILON_START

        self.agent_info = {}

        # loading sequence
        if load == True: self.load()

    def load(self):
        self.net.load_state_dict(torch.load("save_model/net.dat"))
        self.target_net.load_state_dict(torch.load("save_model/target_net.dat"))
        # Note: add loading of epsilon and rest
    
    def save(self):
        torch.save(self.net.state_dict(), "save_model/net.dat")
        torch.save(self.net.state_dict(), "save_model/target_net.dat")
        # Note: add saving of epsilon and rest

    def return_env(self, index=0):
        return self.env_list[index].board
    
    def api(self, index=0):
        return (self.env_list[index].board, self.env_list[index].state, self.epsilon, self.mean_reward, self.env_list[index].steps, self.env_list[index].count_deaths, self.env_list[index].eaten_apples, self.env_list[index].game_info)
    
    def select_device(self):
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            print("using cuda:", torch.cuda.get_device_name(0))
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def calc_loss(self, batch, device="cpu"):
        states, actions, rewards, dones, next_states = batch

        states_v = torch.tensor(states).to(device, dtype=torch.float32)
        next_states_v = torch.tensor(next_states).to(device, dtype=torch.float32)
        actions_v = torch.tensor(actions).to(device, dtype=torch.int64)
        rewards_v = torch.tensor(rewards).to(device, dtype=torch.float32)
        done_mask = torch.ByteTensor(dones).to(device)
        done_mask = done_mask.to(torch.bool)

        state_action_values = self.net.forward(states_v)
        state_action_values = state_action_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_values = self.target_net.forward(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * par.GAMMA + rewards_v
        return self.net.loss_f(state_action_values, expected_state_action_values)
    
    def simulate(self, print_info=True, print_board=False):
        for env_idx, env in enumerate(self.env_list):
            if print_board: 
                print(env_idx)
                print(env.board)

            self.index += 1
            self.epsilon = max(par.EPSILON_FINAL, par.EPSILON_START - self.index / par.EPSILON_DECAY_LAST_FRAME)

            reward = self.play_step(self.net, env, self.epsilon, device=self.device)

            if reward is not None:
                self.total_rewards.append(reward)
                self.mean_reward = np.mean(self.total_rewards[-100:])
                
                if self.best_mean_reward is None or self.best_mean_reward < self.mean_reward:
                    self.save()

                    if self.best_mean_reward is not None:
                        self.agent_info = {"env id": env_idx, "epsilon": self.epsilon, "mean reward": self.mean_reward, "done n of games": (self.index, len(self.total_rewards))}
                        if print_info == True: print(self.agent_info)

                    self.best_mean_reward = self.mean_reward

                if env.game_info["won game"] == True:
                    print("Solved in %d frames!" % self.index)
                    self.save()
                    break

            if len(self.buffer) < par.REPLAY_START_SIZE:
                continue
            
            # After certain amount time target net become first net
            if self.index % par.SYNC_TARGET_LOOPS == 0:
                self.target_net.load_state_dict(self.net.state_dict())

            # Calculate loss of NN and train it
            self.net.optimizer.zero_grad()
            batch = self.buffer.sample(par.BATCH_SIZE)
            loss_t = self.calc_loss(batch, device=self.device)
            loss_t.backward()
            self.net.optimizer.step()