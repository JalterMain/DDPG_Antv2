import gym, torch, random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import statistics
import torch.nn.functional as F
import copy
torch.autograd.set_detect_anomaly(True)

env = gym.make('Ant-v2')
episodes = 10000
gamma = 0.99
loss_fn = nn.MSELoss()
steps = 0
# initialize policy & value network
class PolicyNetwork(nn.Module):

    def __init__(self, beta):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(111, 32), nn.ReLU(),
                                    nn.Linear(32, 32), nn.ReLU(),
                                    nn.Linear(32, 8))
        self.beta = beta
    def forward(self, x):
        if x.shape[0] == 111:
            x = torch.transpose(x, -1, 0)
        return torch.tanh(self.model(x))

    def explore(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        return self.forward(x) + self.beta * torch.randn(8)

class ValueNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(119, 32), nn.ReLU(),
                                    nn.Linear(32, 32), nn.ReLU(),
                                    nn.Linear(32, 1))

    def forward(self, x):
        if x.shape[0] == 119:
            x = torch.transpose(x, -1, 0)
        return self.model(x)

class Buffer():

    def __init__(self, max_size, mini_batch):

        self.buffer = []
        self.max_size = max_size
        self.batch_size = mini_batch

    def add(self, x):
        self.buffer.append(x)
        if len(self.buffer) > self.max_size:
            del self.buffer[0]

    def mini_batch(self):
        return random.sample(self.buffer, self.batch_size)

def extract(x):

    f_tensor = []

    for i in range(5):

        list = [e[i] for e in x]
        if isinstance(list[0], np.ndarray):
            list = [torch.from_numpy(a).float() for a in list]
        if i != 2 and i!=4:
            tensor = torch.stack(tuple(b for b in list))
        elif i == 4:
            tensor = torch.Tensor(list).float().unsqueeze(1)
        else:
            tensor = torch.unsqueeze(torch.tensor(list).float(), 1)
        f_tensor.append(tensor)

    return f_tensor[0], f_tensor[1], f_tensor[2], f_tensor[3], f_tensor[4]

policy = PolicyNetwork(0.2)
value = ValueNetwork()
buffer = Buffer(15000, 256)
target_policy = copy.deepcopy(policy)
target_value = copy.deepcopy(value)
value_optim = optim.Adam(value.parameters(), lr = 0.00001)
policy_optim = optim.Adam(policy.parameters(), lr = 0.000003)
loss_policy = 0
episode = []
loss_value = 0


for e in range(episodes):
    initial_obs = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = (policy.explore(initial_obs)).detach()
        obs, reward, done, _ = env.step(action.numpy()) # make sure the done flag is not time out
        episode_reward += reward
        buffer.add((initial_obs, action, reward, obs, done)) # (s, a, r, s')
        initial_obs = obs

        if len(buffer.buffer) > 1000:

            # training value
            samples = buffer.mini_batch()
            initial_states, actions, rewards, states, finished = extract(samples)
            q_model = value(torch.cat((initial_states, actions), 1))
            q_bellman = rewards + gamma * target_value(torch.cat((states, target_policy(states)),1)) * (1-finished)
            loss_value = loss_fn(q_model, q_bellman.detach())
            loss_value.backward()
            # print([i.grad for i in target_value.parameters()])
            value_optim.step()
            value_optim.zero_grad()
            policy_optim.zero_grad()

            # training policy
            samples = buffer.mini_batch()
            loss_policy = -(value(torch.cat((initial_states, policy.explore(initial_states)),1)).mean())
            loss_policy.backward()
            policy_optim.step()
            value_optim.zero_grad()
            policy_optim.zero_grad()

        steps += 1
        #print(f"Step #{steps}: {-loss_policy}")

        if steps % 10000 == 0:
            target_policy = copy.deepcopy(policy)
            target_value = copy.deepcopy(value)
    print(f"Episode # {e}:{episode_reward}, Steps: {steps}, Loss value: {loss_value}, loss policy: {loss_policy}")