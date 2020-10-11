import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from parameters import *
import environment
import agent

class Neural_Network(nn.Module):
    def __init__(self, lr=LEARNING_RATE):
        super(Neural_Network, self).__init__()
        """
        Input to NN:
            [distance to wall, see apple, see it self, head direction, tail direction] -> 28 elements
        output of NN:
            [0: up    1: right    2: down    3: left] -> 4 elements
        """
        
        self.model = nn.Sequential(
            nn.Linear(INPUT_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, N_ACTIONS)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, input_tensor):
        return self.model(input_tensor)

class SaveAndLoad:
    def load_models(self, net, target_net, agent, device):
        net.load_state_dict(torch.load("model/net.dat", map_location=torch.device(device)))
        target_net.load_state_dict(net.state_dict())
        with open("model/index", 'rb') as f:
            index = np.load(f)[0]
        with open("model/total_rewards", 'rb') as f:
            total_rewards = np.load(f).tolist()
        with open("model/count_deaths", 'rb') as f:
            agent.generation_count = np.load(f)[0]
        return net, target_net, index, total_rewards
    
    def save_models(self, net, agent, index, total_rewards):
        torch.save(net.state_dict(), "model/net.dat")
        with open("model/index", 'wb') as f:
            np.save(f, np.array([index]))
        with open("model/total_rewards", 'wb') as f:
            np.save(f, np.array(total_rewards))
        with open("model/count_deaths", 'wb') as f:
            np.save(f, np.array([agent.generation_count]))

class DQN(SaveAndLoad):
    def __init__(self, net, buffer, agent, load=False):
        #super().__init__()
        self.device = self.select_device()
        self.net = net.to(self.device)
        self.target_net = net.to(self.device)
        self.buffer = buffer
        self.agent = agent

        self.epsilon = EPSILON_START
        self.lr = LEARNING_RATE

        self.second_init(load)
    
    def second_init(self, load):
        # parameters
        self.best_mean_reward = None
        self.mean_reward = None
        self.finished = False
        # loading sequence
        if load: 
            self.net, self.target_net, \
                self.index, self.total_rewards = \
                    self.load_models(self.net, self.target_net, self.agent, self.device)
        else:
            self.index = 0
            self.total_rewards = []

    def select_device(self):
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            print("using cuda:", torch.cuda.get_device_name(0))
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def save(self):
        self.save_models(self.net, self.agent, self.index, self.total_rewards)
    
    def api(self):    
        return (self.agent.env.board, self.agent.env.state, \
                self.epsilon, self.mean_reward, \
                self.agent.env.steps, self.agent.generation_count, \
                self.agent.env.eaten_apples)
    
    def light_api(self):
        return (self.epsilon, self.mean_reward, \
                self.agent.env.steps, self.agent.generation_count, \
                self.agent.env.eaten_apples)
    
    def super_light_api(self):
        return self.agent.env.info
    
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
        state_action_values = self.net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_values = self.target_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * GAMMA + rewards_v
        # Calculate NN loss
        return nn.MSELoss()(state_action_values, expected_state_action_values)
    
    def simulate(self):
        # Training AI
        self.index += 1
        self.epsilon = max(EPSILON_FINAL, EPSILON_START - self.index / EPSILON_DECAY_LAST_FRAME)

        reward = self.agent.play_step(self.net, self.epsilon, device=self.device)

        if reward is not None:
            self.total_rewards.append(reward)
            self.mean_reward = np.mean(self.total_rewards[-100:])
            
            if self.best_mean_reward is None or self.best_mean_reward < self.mean_reward:
                self.save()
                self.total_rewards = self.total_rewards[-100:]

                if self.best_mean_reward is not None:
                    self.agent_info = {"Generation": self.agent.generation_count, "Mean reward": self.mean_reward, "Epsilon": self.epsilon}
                    print(self.agent_info)
                self.best_mean_reward = self.mean_reward

            if self.agent.env.info == "Finished":
                print("Solved in %d frames!" % self.index)
                self.save()
                self.finished = True
                return
            
        if len(self.buffer) < REPLAY_START_SIZE:
            return
        
        # After certain amount time target net become first net
        if self.index % SYNC_TARGET_LOOPS == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        
        # Calculate loss of NN and train it
        self.net.optimizer.zero_grad()
        batch = self.buffer.sample(BATCH_SIZE)
        loss_t = self.calc_loss(batch, device=self.device)
        loss_t.backward()
        self.net.optimizer.step()
    
    def play_env(self, state):
        state_v = torch.tensor(np.array([state], copy=False)).to(self.device, dtype=torch.float32)
        q_vals = self.net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        state, reward, done, _ = self.agent.env.step(action)
        return state