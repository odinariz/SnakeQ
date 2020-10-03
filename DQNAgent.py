import collections
import numpy as np

import parameters as par

import torch
import torch.nn as nn

class Neural_Network(nn.Module):
    def __init__(self, input_size=28, output_size=4, lr=par.LEARNING_RATE):
        super().__init__()
        """
        Input to NN:
            [distance to wall, see apple, see it self, head direction, tail direction] -> 28 elements
        output of NN:
            [0: up    1: right    2: down    3: left] -> 4 elements
        """
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(),
            nn.Linear(20, 12),
            nn.ReLU(),
            nn.Linear(12, output_size),
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
        self.env = env
        self.buffer = buffer
        self.Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])
        self._reset(env)

    def _reset(self, env):
        self.state = env.restart_env()
        self.total_reward_ = 0.0

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
        self.total_reward_ += reward

        exp = self.Experience(self.state, act, reward, is_done, new_state)
        self.buffer.append(exp)
        self.state = new_state

        if is_done:
            done_reward = self.total_reward_
            self._reset(env)
        return done_reward


class DQN(Agent):
    def __init__(self, env, buffer, net, load=True):
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
        self.episode = 0

        self.agent_info = {}

        # loading sequence
        if load: self.load()

    def api(self):
        return (self.env.board, self.env.state, self.epsilon, self.mean_reward, \
            self.env.steps, self.env.count_deaths, self.env.eaten_apples, \
                self.episode, self.env.game_info)
    
    def select_device(self):
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            print("using cuda:", torch.cuda.get_device_name(0))
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self, load_index=True, load_episode=True):
        self.net.load_state_dict(torch.load("save_model/net.dat", map_location=torch.device(self.device)))
        self.target_net.load_state_dict(self.net.state_dict())
        if not load_index:
            with open("save_model/index", 'rb') as f:
                self.index = np.load(f)[0]
        with open("save_model/total_rewards", 'rb') as f:
            self.total_rewards = np.load(f).tolist()
        with open("save_model/count_deaths", 'rb') as f:
            self.env.count_deaths = np.load(f)[0]
        if not load_episode:
            with open("save_model/episode", 'rb') as f:
                self.episode = np.load(f)[0]

    def save(self):
        torch.save(self.net.state_dict(), "save_model/net.dat")
        with open("save_model/index", 'wb') as f:
            np.save(f, np.array([self.index]))
        with open("save_model/total_rewards", 'wb') as f:
            np.save(f, self.total_rewards)
        with open("save_model/count_deaths", 'wb') as f:
            np.save(f, np.array([self.env.count_deaths]))
        with open("save_model/episode", 'wb') as f:
            np.save(f, np.array([self.episode]))
    
    def calc_loss(self, batch, device="cpu"):
        # unpack batch
        states, actions, rewards, dones, next_states = batch

        # convert everything from batch to torch tensors and move it to device
        states_v = torch.tensor(states).to(device, dtype=torch.float32)
        next_states_v = torch.tensor(next_states).to(device, dtype=torch.float32)
        actions_v = torch.tensor(actions).to(device, dtype=torch.int64)
        rewards_v = torch.tensor(rewards).to(device, dtype=torch.float32)
        done_mask = torch.ByteTensor(dones).to(device)
        done_mask = done_mask.to(torch.bool)

        # get output from NNs which is used for calculating state action value with discount
        state_action_values = self.net.forward(states_v)
        state_action_values = state_action_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_values = self.target_net.forward(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * par.GAMMA + rewards_v
        # Calculate NN loss
        return self.net.loss_f(state_action_values, expected_state_action_values)
    
    def simulate(self, print_info=True):
        # Training AI

        self.index += 1
        self.epsilon = max(par.EPSILON_FINAL, par.EPSILON_START - self.index / par.EPSILON_DECAY_LAST_FRAME)

        reward = self.play_step(self.net, self.env, self.epsilon, device=self.device)

        if reward is not None:

            self.total_rewards.append(reward)
            self.mean_reward = np.mean(self.total_rewards[-100:])
            
            if self.best_mean_reward is None or self.best_mean_reward < self.mean_reward:
                self.save()

                if self.best_mean_reward is not None:
                    self.agent_info = {"Generation": self.env.count_deaths, "Mean reward": self.mean_reward, "Epsilon": self.epsilon, "Episode": self.episode}
                    if print_info: print(self.agent_info)

                self.best_mean_reward = self.mean_reward

            if self.env.game_info["won game"]:
                print("Solved in %d frames!" % self.index)
                self.save()
                return
            
            """
            if self.best_mean_reward > self.mean_reward and self.epsilon < 0.1:
                self.episode += 1
                self.load(load_index=False, load_episode=False)
            """

        if len(self.buffer) < par.REPLAY_START_SIZE:
            return
        
        # After certain amount time target net become first net
        if self.index % par.SYNC_TARGET_LOOPS == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        # Calculate loss of NN and train it
        self.net.optimizer.zero_grad()
        batch = self.buffer.sample(par.BATCH_SIZE)
        loss_t = self.calc_loss(batch, device=self.device)
        loss_t.backward()
        self.net.optimizer.step()