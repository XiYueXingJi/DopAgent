"""
=============================================================================
DopAgent —— 多巴胺增强的 DQN 智能体 (最终版)

核心创新：
  1. 前向模型预测误差 = 内在奖赏（好奇心信号）
  2. 内在奖赏动态重算（不用过期值）
  3. 好奇心直接影响动作选择（不只是事后奖励）

R_total = R_ext + beta * R_int
R_int = MSE(f_forward(state, action), next_state)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dqn_agent import QNetwork, ReplayBuffer
from dopamine_model import DopamineModule


class DopAgent:
    def __init__(self, state_dim, action_dim, beta=1.0,
                 lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 memory_size=10000, batch_size=64, target_update=100,
                 forward_lr=1e-3, forward_hidden=[128, 64], device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.beta = beta
        self.device = device

        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(memory_size)

        self.dopamine = DopamineModule(
            state_dim, action_dim, lr=forward_lr,
            hidden_dims=forward_hidden, device=device)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.train_steps = 0

        self.episode_rewards = []
        self.episode_ext_rewards = []
        self.episode_int_rewards = []

    def select_action(self, state, eval_mode=False):
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return q_values.argmax().item()

    def update(self, state, action, ext_reward, next_state, done):
        # Step 1: 内在奖赏
        r_int = self.dopamine.compute_intrinsic_reward(state, action, next_state)
        total_reward = ext_reward + self.beta * r_int

        # Step 2: 存经验（只存 ext_reward！）
        self.memory.push(state, action, ext_reward, next_state, done)

        if len(self.memory) < self.batch_size:
            return None, {'r_int': r_int, 'r_total': total_reward}

        # Step 3: 采样 + 动态重算内在奖赏
        states, actions, ext_rewards, next_states, dones = \
            self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        ext_rewards = ext_rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 用当前最新前向模型重算内在奖赏
        intrinsic_rewards = []
        for i in range(self.batch_size):
            s_i = states[i].cpu().numpy()
            a_i = actions[i].item()
            ns_i = next_states[i].cpu().numpy()
            r_int_i = self.dopamine.compute_intrinsic_reward(s_i, a_i, ns_i)
            intrinsic_rewards.append(r_int_i)
        intrinsic_rewards = torch.FloatTensor(intrinsic_rewards).to(self.device)
        total_rewards = ext_rewards + self.beta * intrinsic_rewards

        # Step 4: DQN 更新
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            targets = total_rewards + self.gamma * next_q * (1 - dones)

        current_q = self.q_network(states).gather(
            1, actions.unsqueeze(1)).squeeze()
        loss = F.mse_loss(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Step 5: 更新前向模型
        forward_loss = self.dopamine.update(state, action, next_state)

        return loss.item(), {
            'r_int': r_int, 'r_total': total_reward,
            'forward_loss': forward_loss}

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def record_episode(self, ext_reward, int_reward, total_reward):
        self.episode_ext_rewards.append(ext_reward)
        self.episode_int_rewards.append(int_reward)
        self.episode_rewards.append(total_reward)

    def save(self, path):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_ext_rewards': self.episode_ext_rewards,
            'episode_int_rewards': self.episode_int_rewards,
        }, path)
        self.dopamine.save(path.replace('.pth', '_dopamine.pth'))

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(ckpt['q_network'])
        self.target_network.load_state_dict(ckpt['target_network'])
        self.epsilon = ckpt['epsilon']
        self.episode_rewards = ckpt['episode_rewards']
        self.episode_ext_rewards = ckpt['episode_ext_rewards']
        self.episode_int_rewards = ckpt['episode_int_rewards']
        self.dopamine.load(path.replace('.pth', '_dopamine.pth'))
