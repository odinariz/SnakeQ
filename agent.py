import collections
import numpy as np
import torch

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

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
    def __init__(self, env, exp_buffer):
        self.env = env
        self.reset_env = env
        self.exp_buffer = exp_buffer
        self._reset()
        self.generation_count = 0

    def _reset(self):
        #self.state = self.env.reset()
        self.state = self.reset_env.reset()
        self.total_reward = 0.0
    
    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:    # select random action or action from NN
            action = self.env.sample_action()
        else:
            state_a = np.array([self.state], copy=False).astype("float32")
            state_v = torch.from_numpy(state_a).to(device, dtype=torch.float32)
            q_vals_v = net(state_v)
            _, action_v = torch.max(q_vals_v, dim=1)
            action = int(action_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state

        if is_done:
            done_reward = self.total_reward
            self._reset()
            self.generation_count += 1
        return done_reward