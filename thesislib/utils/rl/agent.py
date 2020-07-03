import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
from collections import namedtuple, OrderedDict

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, input_dim, output_dim, layer_config=None, non_linearity='relu'):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_config = self.form_layer_config(layer_config)
        self.non_linearity = non_linearity
        self.model = self.compose_model()

    def get_default_config(self):
        return [
            [self.input_dim, 48],
            [48, 32],
            [32, 32],
            [32, self.output_dim]
        ]

    def form_layer_config(self, layer_config):
        if layer_config is None:
            return self.get_default_config()

        if len(layer_config) < 2:
            raise ValueError("Layer config must have at least two layers")

        if layer_config[0][0] != self.input_dim:
            raise ValueError("Input dimension of first layer config must be the same as input to the model")

        if layer_config[-1][1] != self.output_dim:
            raise ValueError("output dimension of last layer config must be the same as expected model output")

        for idx in range(len(layer_config) - 1):
            assert layer_config[idx][1] == layer_config[idx+1][0], "Dimension mismatch between layers %d and %d" % (idx, idx + 1)

        return layer_config

    def get_non_linear_class(self):
        if self.non_linearity == 'tanh':
            return nn.Tanh
        else:
            return nn.ReLU

    def compose_model(self):
        non_linear = self.get_non_linear_class()
        layers = OrderedDict()
        for idx in range(len(self.layer_config)):
            input_dim, output_dim = self.layer_config[idx]
            layers['linear-%d' % idx] = nn.Linear(input_dim, output_dim)
            if idx != len(self.layer_config) - 1:
                layers['nonlinear-%d' % idx] = non_linear()

        return nn.Sequential(layers)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        return self.model(x)


class MedAgent:
    def __init__(self, env, **kwargs):
        self.env = env
        # 3 takes care of the age, gender and race and 3*num_symptoms represents the symptoms flattened out
        self.input_dim = 3 + 3*env.num_symptoms
        self.output_dim = env.num_symptoms + env.num_conditions
        self.n_actions = self.output_dim
        self.layer_config = kwargs.get('layer_config', None)
        self.learning_start  = kwargs.get('learning_start', 1)
        self.batch_size = kwargs.get('batch_size', 1)
        self.gamma = kwargs.get('gamma', 0.999)
        self.eps_start = kwargs.get('eps_start', 0.9)
        self.epsilon = self.eps_start
        self.eps_end = kwargs.get('eps_end', 0.05)
        self.eps_decay = kwargs.get('eps_decay', 200)
        self.target_update = kwargs.get('target_update', 10)
        self.replay_capacity = kwargs.get('replay_capacity', 10)
        self.non_linearity = kwargs.get('non_linearity', 'relu')
        self.optimiser_name = kwargs.get('optimiser_name', 'rmsprop')
        self.optimiser_params = kwargs.get('optimiser_params', {})
        self.debug = kwargs.get('debug', False)

        self.memory = ReplayMemory(self.replay_capacity)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps_done = 0

        self.policy_network = DQN(self.input_dim, self.output_dim, self.layer_config, self.non_linearity).to(
            self.device)
        self.target_network = DQN(self.input_dim, self.output_dim, self.layer_config, self.non_linearity).to(
            self.device)

        # we aren't interested in tracking gradients for the target network
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        optimiser_cls = self.get_optimiser()
        self.optimiser = optimiser_cls(self.policy_network.parameters(), **self.optimiser_params)

        self.state = None
        self.reset_env()

    def reset_env(self):
        self.env.reset()
        self.state = self.state_to_tensor(self.env.state)

    def state_to_tensor(self, state):
        if state is None:
            return None

        tensor = np.zeros(self.input_dim)

        tensor[0] = state.gender
        tensor[1] = state.race
        tensor[2] = state.age

        tensor[3:] = state.symptoms.reshape(-1)

        return torch.tensor(tensor, device=self.device, dtype=torch.float).reshape(-1, self.input_dim)

    def get_optimiser(self):
        if self.optimiser_name == 'sgd':
            optimiser = optim.RMSprop
        elif self.optimiser_name == 'adam':
            optimiser = optim.Adam
        else:
            optimiser = optim.RMSprop

        return optimiser

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        if self.steps_done < self.learning_start:
            return None

        transitions = self.memory.sample(self.batch_size)

        # see https://stackoverflow.com/a/19343/3343043 for detailed explanation
        batch = Transition(*zip(*transitions))

        loss = self.compute_loss(batch)

        self.optimiser.zero_grad()
        loss.backward()

        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimiser.step()

        return loss

    def double_q_update(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def compute_loss(self, batch):
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_network.forward(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        next_state_values[non_final_mask] = self.target_network.forward(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        return loss

    def select_action(self, state):
        sample = random.random()

        if sample < self.epsilon:
            if self.debug:
                print("Taking Random Action")
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
        else:
            if self.debug:
                print("Taking Best Action")
            with torch.no_grad():
                return self.policy_network.forward(state).max(1)[1].view(1, 1)

    def decay_epsilon(self):
        if self.steps_done > self.learning_start:
            self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
                           np.exp(-1 * (self.steps_done-self.learning_start) / self.eps_decay)

    def step(self):
        state = self.state
        action = self.select_action(state)

        if self.debug:
            print("State: ", state)
            print("Took action: ", action)

        self.steps_done += 1

        self.decay_epsilon()

        next_state, _reward, done = self.env.take_action(action.item())
        if self.debug:
            print("Next state: ", next_state)
            print("Reward: ", _reward)
            print("Done: ", done)
        next_state = self.state_to_tensor(next_state)
        reward = torch.tensor([_reward], device=self.device)

        self.memory.push(state, action, next_state, reward)
        self.state = next_state

        if self.steps_done % self.target_update == 0:
            self.double_q_update()

        return _reward, done

    def __del__(self):
        del self.env
