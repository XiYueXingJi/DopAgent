"""
=============================================================================
基线 DQN 智能体 (Baseline Deep Q-Network)

这是成员B的对照组实现：
- 标准 DQN：Q-Network + Target Network + Experience Replay
- 在稀疏奖赏环境中，这个智能体将表现极差
- 它需要"看到"奖励才能学习，而迷宫只在终点有奖励
- 结果：智能体只在起点附近随机游走，永远找不到终点

这恰恰证明了"仅有外部奖赏是不够的"——需要内在动机（好奇心）！
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class QNetwork(nn.Module):
    """
    深度Q网络

    输入状态 s (2维坐标)，输出每个动作的 Q 值
    架构：简单的 MLP (多层感知机)
    """

    def __init__(self, state_dim, action_dim, hidden_dims=[128, 64]):
        """
        参数:
            state_dim: 状态维度 (本项目中 = 2)
            action_dim: 动作维度 (本项目中 = 4)
            hidden_dims: 隐藏层维度列表
        """
        super().__init__()
        layers = []
        input_dim = state_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim

        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """
        前向传播

        参数:
            state: [batch_size, state_dim] 状态张量

        返回:
            q_values: [batch_size, action_dim] 每个动作的Q值
        """
        return self.network(state)


class ReplayBuffer:
    """
    经验回放缓冲区

    存储 (state, action, reward, next_state, done) 五元组
    训练时随机采样，打破数据间的时序相关性
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """存入一条经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, batch_size)

        states = torch.FloatTensor(np.array([t[0] for t in batch]))
        actions = torch.LongTensor(np.array([t[1] for t in batch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in batch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in batch]))
        dones = torch.FloatTensor(np.array([t[4] for t in batch]))

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    基线 DQN 智能体

    只使用外部奖赏 (extrinsic reward) 来学习。
    在稀疏奖赏迷宫中，这注定失败——因为没有中间奖励引导。

    认知科学视角：
    这就像一个缺乏多巴胺能神经元的人/动物——没有"想探索"的内在驱动力，
    只有获得真实奖励时才能学习。在复杂环境中，这几乎等于学不到任何东西。
    """

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 memory_size=10000, batch_size=64, target_update=100,
                 device='cpu'):
        """
        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            lr: 学习率
            gamma: 折扣因子
            epsilon_start/end/decay: 探索率参数
            memory_size: 经验池大小
            batch_size: 批量采样大小
            target_update: 目标网络更新频率
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device

        # 主网络和目标网络
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 优化器
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # 经验回放池
        self.memory = ReplayBuffer(memory_size)

        # 步数计数器（用于目标网络更新）
        self.train_steps = 0

        # 跟踪探索情况（用于分析）
        self.episode_rewards = []

    def select_action(self, state, eval_mode=False):
        """
        选择动作：epsilon-greedy 策略

        参数:
            state: 当前状态
            eval_mode: True=纯贪心（评估模式），False=探索模式

        返回:
            action: 选择的动作 (0-3)
        """
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def update(self, state, action, reward, next_state, done):
        """
        单步更新：存经验 → 从经验池采样 → 训练

        参数:
            state, action, reward, next_state, done: 当前转移

        返回:
            loss: TD误差 (MSE loss)，如果经验不够则返回 None
        """
        # 存入经验池
        self.memory.push(state, action, reward, next_state, done)

        # 经验不够，不训练
        if len(self.memory) < self.batch_size:
            return None

        # 采样
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # === DQN 核心更新公式 ===
        # 目标: y = r + γ * max_a' Q_target(s', a')  (如果 done，则只有 r)
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q = next_q_values.max(dim=1)[0]
            targets = rewards + self.gamma * max_next_q * (1 - dones)

        # 当前 Q 值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # TD误差 (Temporal Difference Error)
        loss = F.mse_loss(current_q, targets)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        self.train_steps += 1
        if self.train_steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.epsilon = checkpoint['epsilon']

    def record_episode(self, total_reward):
        """记录每轮的总奖励"""
        self.episode_rewards.append(total_reward)


# ==================== 测试代码 ====================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'g:/code/rengongzhinengdaolun/naoyurenzhi')
    from environment import GridWorld

    print("=== 快速测试：DQN Agent 与环境交互 ===\n")

    env = GridWorld(size=8, max_steps=30, seed=42)
    agent = DQNAgent(state_dim=2, action_dim=4, device='cpu')

    state = env.reset()
    total_reward = 0

    for step in range(30):
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        loss = agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if loss is not None:
            print(f"步{step:2d}: 动作={action}, 位置={env.agent_pos}, "
                  f"奖励={reward:.2f}, Loss={loss:.4f}")
        else:
            print(f"步{step:2d}: 动作={action}, 位置={env.agent_pos}, "
                  f"奖励={reward:.2f}, (经验收集中...)")

        if done:
            break

    print(f"\n总奖励: {total_reward:.2f}")
    print(f"经验池大小: {len(agent.memory)}")
    print("DQN Agent 基本功能正常！")
