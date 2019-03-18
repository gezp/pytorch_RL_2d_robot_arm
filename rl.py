import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
directory='model/'

DDPG_CONFIG={
'LR_ACTOR':0.001,
'LR_CRITIC':0.001,
'GAMMA':0.9,
'TAU':0.01,
'MEMORY_CAPACITY':10000,
'BATCH_SIZE':32,

}

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400 , 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.memory = np.zeros((DDPG_CONFIG['MEMORY_CAPACITY'], state_dim * 2 + action_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False
        self.state_dim=state_dim
        self.action_dim=action_dim

        self.actor = Actor(state_dim, action_dim, max_action[1]).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action[1]).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), DDPG_CONFIG['LR_ACTOR'])

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), DDPG_CONFIG['LR_CRITIC'])
 


    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def store_transition(self, states, actions, rewards, states_next):
        transitions = np.hstack((states, actions, [rewards], states_next))
        index = self.pointer % DDPG_CONFIG['MEMORY_CAPACITY']
        #print("size:",states.size,actions.size,states_next.size)
        self.memory[index, :] = transitions
        self.pointer += 1
        if self.pointer > DDPG_CONFIG['MEMORY_CAPACITY']:      # indicator for learning
            self.memory_full = True

    def learn(self):
        indices = np.random.choice(DDPG_CONFIG['MEMORY_CAPACITY'], size = DDPG_CONFIG['BATCH_SIZE'])
        bt = torch.Tensor(self.memory[indices, :])
        state = bt[:, :self.state_dim].to(device)
        action = bt[:, self.state_dim: self.state_dim + self.action_dim].to(device)
        reward = bt[:, -self.state_dim - 1: -self.state_dim].to(device)
        next_state = bt[:, -self.state_dim:].to(device)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (DDPG_CONFIG['GAMMA'] * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(DDPG_CONFIG['TAU'] * param.data + (1 - DDPG_CONFIG['TAU']) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(DDPG_CONFIG['TAU'] * param.data + (1 - DDPG_CONFIG['TAU']) * target_param.data)


    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def restore(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")