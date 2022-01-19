import gym, torch, random, copy
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import neptune.new as neptune

# initialize policy & value network
class PolicyNetwork(nn.Module):

    def __init__(self, beta):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(111, 400), nn.ReLU(),
                                    nn.Linear(400, 300), nn.ReLU(),
                                    nn.Linear(300, 8))
        self.beta = beta
    def forward(self, x):
        return torch.tanh(self.model(x))

    def explore(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        return self.forward(x) + self.beta * torch.randn(8)

class ValueNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(119, 400), nn.ReLU(),
                                    nn.Linear(400, 300), nn.ReLU(),
                                    nn.Linear(300, 1))

    def forward(self, x):
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

# This is messy
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

class TrainingLoop():
    def __init__(self, policy_net, value_net, buffer, value_optim, policy_optim, episodes = 1000, gamma = 0.99, loss_fn = nn.MSELoss(), env=gym.make('Ant-v2'), k = 2, min_buffer_size = 1000, tau = 0.0005):
        self.policy_net = policy_net
        self.value_net = value_net
        self.target_policy = copy.deepcopy(self.policy_net)
        self.target_value = copy.deepcopy(self.value_net)
        self.buffer = buffer
        self.value_optim = value_optim
        self.policy_optim = policy_optim
        self.episodes = episodes
        self.gamma = gamma
        self.loss_fn = loss_fn
        self.extract_fn = extract
        self.episode_reward = 0
        self.env = env
        self.k = k
        self.run = neptune.init(project = "jaltermain/DDPGAntv-2")
        self.initial_obs = None
        self.val_losses = []
        self.pol_losses = []
        self.min_buffer_size = min_buffer_size
        self.done = False
        self.steps = 0
        self.tau = tau

    def interact(self):
        action = (self.policy_net.explore(self.initial_obs)).detach()
        obs, reward, done, _ = self.env.step(action.numpy()) # make sure the done flag is not time out
        self.episode_reward += reward
        self.buffer.add((self.initial_obs, action, reward, obs, done)) # (s, a, r, s')
        self.initial_obs = obs
        self.done = done

    def get_batch(self):
        samples = self.buffer.mini_batch()
        return self.extract_fn(samples)

    def train_value(self):
        initial_states, actions, rewards, states, finished = self.get_batch()
        q_model = value(torch.cat((initial_states, actions), 1))
        q_bellman = rewards + self.gamma * self.target_value(torch.cat((states, self.target_policy(states)),1)) * (1-finished)
        loss_value = self.loss_fn(q_model, q_bellman.detach())
        self.val_losses.append(loss_value.detach())
        # run['train/value_loss'].log(loss_value)
        loss_value.backward()
        # print([i.grad for i in target_value.parameters()])
        self.value_optim.step()
        self.value_optim.zero_grad()
        self.policy_optim.zero_grad()

    def train_policy(self):
        initial_states, actions, rewards, states, finished = self.get_batch()
        loss_policy = -(self.value_net(torch.cat((initial_states, self.policy_net.explore(initial_states)),1)).mean())
        self.pol_losses.append(loss_policy.detach())
        # run['train/policy_loss'].log(loss_policy)
        loss_policy.backward()
        self.policy_optim.step()
        self.value_optim.zero_grad()
        self.policy_optim.zero_grad()

    def polyak_averaging(self):
        params1 = self.value_net.named_parameters()
        params2 = self.target_value.named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(self.tau*param1.data + (1-self.tau)*dict_params2[name1].data)
        if self.steps % self.k == 0:
            params1 = self.policy_net.named_parameters()
            params2 = self.target_policy.named_parameters()

            dict_params2 = dict(params2)

            for name1, param1 in params1:
                if name1 in dict_params2:
                    dict_params2[name1].data.copy_(self.tau*param1.data + (1-self.tau)*dict_params2[name1].data)

    def init_log(self):
        network_hyperparams = {'Optimizer': 'Adam','Value-learning_rate': 0.0001, 'Policy-learning_rate': 0.0001, 'loss_fn_value': 'MSE'}
        self.run["network_hyperparameters"] = network_hyperparams
        network_sizes = {'ValueNet_size': '(119,400,300,1)', 'PolicyNet_size': '(111,400,300,8)'}
        self.run["network_sizes"] = network_sizes
        buffer_params = {'buffer_maxsize': self.buffer.max_size, 'batch_size': self.buffer.batch_size, 'min_size_train': self.min_buffer_size}
        self.run['buffer_parameters'] = buffer_params
        policy_params = {'exploration_noise': policy.beta, 'policy_smoothing': policy.beta}
        self.run['policy_parameters'] = policy_params
        environment_params = {'gamma': self.gamma, 'env_name': 'Ant-v2', 'episodes': self.episodes}
        self.run['environment_parameters'] = environment_params

        self.run['environment_parameters'] = environment_params
    def log(self):
        self.run['train/episode_reward'].log(self.episode_reward)
        if not self.val_losses:
            self.run['train/mean_episodic_value_loss'].log(0)
            self.run['train/mean_episodic_policy_loss'].log(0)
        else:
            mean_val_loss = sum(self.val_losses) / len(self.val_losses)
            mean_pol_loss = sum(self.pol_losses) / len(self.pol_losses)
            self.run['train/mean_episodic_value_loss'].log(mean_val_loss)
            self.run['train/mean_episodic_policy_loss'].log(mean_pol_loss)

    def training_loop(self):
        self.init_log()
        for e in range(self.episodes):
            self.episode_reward = 0
            self.done = False
            self.val_losses, self.pol_losses = [], []
            self.initial_obs = self.env.reset()
            while not self.done:
                self.interact()
                if len(self.buffer.buffer) > self.min_buffer_size:
                    self.train_value()
                    if self.steps % self.k == 0:
                        self.train_policy()
                self.polyak_averaging()
                self.steps += 1
            self.log()
        self.run.stop()

policy = PolicyNetwork(0.1)
value = ValueNetwork()
buffer = Buffer(15000, 128)
value_optim = optim.Adam(value.parameters(), lr = 0.0001)
policy_optim = optim.Adam(policy.parameters(), lr = 0.0001)

training_loop = TrainingLoop(policy, value, buffer, value_optim, policy_optim)
training_loop.training_loop()
