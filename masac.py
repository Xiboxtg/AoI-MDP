import os
import json
import numpy as np
import random
import time
import sys
from collections import deque
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
#from torch.utils.tensorboard import SummaryWriter
import gc
import torch.nn as nn
import math
#import tensorboard
from collections import deque
import copy
from torch.optim import Adam
import pickle
#import seaborn as sns
import matplotlib.pyplot as plt
import warnings



class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state,action,reward,next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state,action,reward,next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        state,action,reward,next_state, done = map(np.stack, zip(*batch))
        return state,action,reward,next_state, done
    
    def __len__(self):
        return len(self.buffer)

replay_buffer1 = ReplayBuffer(80000)
replay_buffer2 = ReplayBuffer(80000)

def store_data(numb):
    if numb == 1:
        with open('replay_buffer1.pkl', 'wb') as f:
            pickle.dump(replay_buffer1, f)

    if numb == 2:
        with open('replay_buffer2.pkl', 'wb') as f:
            pickle.dump(replay_buffer2, f)


def store_transition(state,action,reward,next_state, done=False,numb=1):
    if numb == 1:
        replay_buffer1.push(state,action,reward,next_state, done)
    elif numb == 2:
        replay_buffer2.push(state,action,reward,next_state, done)
    pass

def is_training(numb=1):
    if numb == 1:
        return len(replay_buffer1) > 5000
    elif numb == 2:
        return len(replay_buffer2) > 5000

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        
        # Q1
        self.linear1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4_q1 = nn.Linear(hidden_dim, 1)
        
        # Q2
        self.linear1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4_q2 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)
        
    def forward(self, state, action):
        x_state_action = torch.cat([state,action], 1)
        
        x1 = F.relu(self.linear1_q1(x_state_action))
        x1 = F.relu(self.linear2_q1(x1))
        x1 = F.relu(self.linear3_q1(x1))
        x1 = self.linear4_q1(x1)
        
        x2 = F.relu(self.linear1_q2(x_state_action))
        x2 = F.relu(self.linear2_q2(x2))
        x2 = F.relu(self.linear3_q2(x2))
        x2 = self.linear4_q2(x2)
        
        return x1, x2



class PolicyNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)

        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.mean_linear_z = nn.Linear(hidden_dim, 1)
        self.log_std_linear_z = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        z_mean = self.mean_linear_z(x)
        z_log_std = self.log_std_linear_z(x)
        z_log_std = torch.clamp(z_log_std, min=self.log_std_min, max=self.log_std_max)
        #z = self.z_linear(x)
        return mean, log_std, z_mean, z_log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std, z_mean, z_log_std = self.forward(state)
        std = log_std.exp()
        z_std = z_log_std.exp()
        # ref:  `from torch.distributions import Normal`
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)# tanh
        z_normal = Normal(z_mean, z_std)
        z_t = z_normal.rsample()
        z = torch.sigmoid(z_t) * 2.5

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + epsilon) # => tanh
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, mean, log_std, z




#  -----------------   SAC   ----------------------
class SAC(object):
    def __init__(self, state_dim,
                 action_dim, gamma=0.99, 
                 tau=1e-2, 
                 alpha=0.4,
                 hidden_dim=250,
                 lr=0.0003,
                 numb=1):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lr=lr
        self.target_update_interval = 1
        self.numb = numb

        self.q_loss =0.0
        self.policy_loss=0.0
        self.alpha_loss =0.0
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        self.device = "cuda:0"




        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        hard_update(self.critic_target, self.critic)



        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
        print('entropy', self.target_entropy)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device) # 优化alpha的对数，符合约束。
        self.alpha_optim = Adam([self.log_alpha], lr=self.lr)


        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)



    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _, _, z = self.policy.sample(state)
        else:
            _, _, action, _, z = self.policy.sample(state)
            action = torch.tanh(action)
        action = action.detach().cpu().numpy()[0]
        return action, z


    def update_parameters(self, batch_size,numb=1):
        memory = replay_buffer1 if numb == 1 else replay_buffer2


        state_batch, action_batch,reward_batch,next_state_batch,done_batch = memory.sample(batch_size=batch_size) # 采样

        # numpy -> tensor
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1) # 奖励也需要变成向量。

        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():

            next_state_action, next_state_log_pi, _, _, z = self.policy.sample(next_state_batch)

            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch,next_state_action)

            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * (min_qf_next_target)#TD target

        # state_batch = torch.cat([state1_batch,state2_batch], 1)
        # action_batch = torch.cat([action1_batch,action2_batch], 1)、

        qf1, qf2 = self.critic(state_batch,action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value) #
        qf2_loss = F.mse_loss(qf2, next_q_value) #

        qf_loss = qf1_loss + qf2_loss
        self.q_loss =qf_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()




        pi, log_pi, mean, log_std, z = self.policy.sample(state_batch)
        # pi = torch.cat([pi1,pi2], 1)
        qf1_pi, qf2_pi = self.critic(state_batch,pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        #policy_loss2 = ((self.alpha * log_pi2) - min_qf_pi).mean()
        self.policy_loss =policy_loss
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()


        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_loss =alpha_loss
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()
        soft_update(self.critic_target, self.critic, self.tau)




        #Save models.
    def save_models(self, model_path,episode_count):
        model_path = os.getcwd() + model_path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.policy.state_dict(),'{}{}_actor_eps{}.pth'.format(model_path,self.numb,str(episode_count)))
        torch.save(self.critic.state_dict(),'{}{}_critic_eps{}.pth'.format(model_path,self.numb,str(episode_count)))
        torch.save(self.critic_target.state_dict(),'{}{}_critic_target_eps{}.pth'.format(model_path,self.numb,str(episode_count)))
        torch.save(self.log_alpha,'{}{}_alpha_eps{}.pth'.format(model_path,self.numb,str(episode_count)))


    #Load models.
    def load_models(self,version, episode):
        model_path = os.getcwd() + '/{}/{}/'.format(version, 'models_m2')
        self.policy.load_state_dict(torch.load('{}{}_actor_eps{}.pth'.format(model_path,self.numb,str(episode))))
        self.critic.load_state_dict(torch.load('{}{}_critic_eps{}.pth'.format(model_path,self.numb,str(episode))))
        self.critic_target.load_state_dict(torch.load('{}{}_critic_target_eps{}.pth'.format(model_path,self.numb,str(episode))))
        self.log_alpha = torch.load('{}{}_alpha_eps{}.pth'.format(model_path,self.numb,str(episode)))
        #hard_update(self.critic_target, self.critic)
        self.alpha_optim = Adam([self.log_alpha],lr=self.lr)
        self.alpha = self.log_alpha.exp()

        self.policy.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)
        self.log_alpha.to(self.device)
        print('***Models load***')


