"""
=============================================================================
DopAgent —— 多巴胺驱动的好奇心强化学习智能体 (核心创新算法)

这是成员C的"技术灵魂"代码！

算法原理（一句话版本）：
    在标准 DQN 的基础上，增加一个"多巴胺前向预测网络"，
    用预测误差（MSE）作为内在奖赏，驱动智能体主动探索未知区域。

总奖赏公式：
    R_total = R_ext + β × R_int

    其中:
    - R_ext: 外部奖赏（环境给的，本项目只在终点=1，其余=0）
    - R_int: 内在奖赏（预测误差，MSE(predicted_next, actual_next)）
    - β: 多巴胺调节系数（控制好奇心强度）

认知科学对应：
    - R_ext → 多巴胺能神经元的时相性发放（实际奖励）
    - R_int → 多巴胺能神经元的"新奇性"发放（预测误差 = RPE）
    - β → 多巴胺受体的敏感度（个体差异）
=============================================================================
"""

import torch
import numpy as np
from dqn_agent import DQNAgent, QNetwork, ReplayBuffer
from dopamine_model import DopamineModule


class DopAgent:
    """
    多巴胺增强的 DQN 智能体

    与基线 DQN 的唯一区别（也是核心创新！）：
    在环境奖励的基础上，加上前向模型的预测误差作为"内在奖赏"。
    这使得智能体即使在稀疏奖赏环境中，也能通过"好奇心"进行探索学习。

    类比：就像婴儿学习走路——不需要外部奖励（糖），内在的好奇心驱使他们
          不断尝试、探索新动作、学习预测结果。
    """

    def __init__(self, state_dim, action_dim,
                 beta=0.5,                   # 多巴胺系数！
                 lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 memory_size=10000, batch_size=64, target_update=100,
                 forward_lr=1e-3, forward_hidden=[128, 64],
                 device='cpu'):
        """
        参数说明与基线DQN相同，新增:
            beta: 内在奖赏系数（"好奇心"有多强？）
            forward_lr: 前向模型的学习率
            forward_hidden: 前向模型隐藏层
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.beta = beta
        self.device = device

        # 继承标准 DQN 的所有组件
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(memory_size)

        # === 核心创新：多巴胺前向预测模块 ===
        # 这就是我们项目最独特的部分！
        self.dopamine = DopamineModule(
            state_dim, action_dim, lr=forward_lr,
            hidden_dims=forward_hidden, device=device
        )

        # DQN 参数
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.train_steps = 0

        # 记录
        self.episode_rewards = []       # 总奖赏历史
        self.episode_ext_rewards = []   # 外部奖赏历史
        self.episode_int_rewards = []   # 内在奖赏历史

    def select_action(self, state, eval_mode=False):
        """epsilon-greedy 动作选择（同基线DQN）"""
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def update(self, state, action, ext_reward, next_state, done):
        """
        单步更新 —— 这里多了关键的内在奖赏计算！

        流程:
            1. 用多巴胺模块计算内在奖赏 (R_int)
            2. 融合: R_total = R_ext + β × R_int
            3. 用 R_total 更新 DQN（标准流程）
            4. 更新多巴胺模块（让它预测越来越准 → 习惯化）

        参数:
            state, action, ext_reward, next_state, done: 环境转移
                ext_reward 是环境给的外部奖赏（稀疏的！）

        返回:
            loss: DQN的TD误差，如果经验不够返回None
            info: 包含 R_int, R_total 等信息的字典
        """
        # ==========================================
        # ✨ 核心创新 Step 1: 计算内在奖赏！
        # ==========================================
        # 前向模型预测 "执行action后会到达什么状态"
        # 预测误差 = 新奇程度 = 内在奖赏
        r_int = self.dopamine.compute_intrinsic_reward(state, action, next_state)

        # ==========================================
        # ✨ 核心创新 Step 2: 融合奖赏！
        # ==========================================
        total_reward = ext_reward + self.beta * r_int

        # ==========================================
        # Step 3: 存入经验池 — ⚠️ 只存外部奖赏！
        # 内在奖赏在训练时动态重算（因为前向模型在不断学习）
        # ==========================================
        self.memory.push(state, action, ext_reward, next_state, done)

        # 经验不够，不训练
        if len(self.memory) < self.batch_size:
            return None, {'r_int': r_int, 'r_total': total_reward}

        # ==========================================
        # Step 4: DQN 更新 —— 关键修复！
        # 从经验池采样后，用当前前向模型重新计算内在奖赏
        # ==========================================
        states, actions, ext_rewards, next_states, dones = self.memory.sample(
            self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        ext_rewards = ext_rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # ✨ 用当前最新的前向模型重算内在奖赏！
        # 这才是正确的：前向模型 → 内在奖赏 → 驱动探索
        intrinsic_rewards = []
        for i in range(self.batch_size):
            s_i = states[i].cpu().numpy()
            a_i = actions[i].item()
            ns_i = next_states[i].cpu().numpy()
            r_int_i = self.dopamine.compute_intrinsic_reward(s_i, a_i, ns_i)
            intrinsic_rewards.append(r_int_i)
        intrinsic_rewards = torch.FloatTensor(intrinsic_rewards).to(self.device)

        # 总奖赏 = 外部 + β × 内在（用最新的前向模型计算！）
        total_rewards = ext_rewards + self.beta * intrinsic_rewards

        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            targets = total_rewards + self.gamma * next_q * (1 - dones)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = torch.nn.functional.mse_loss(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        self.train_steps += 1
        if self.train_steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # ==========================================
        # Step 5: 更新多巴胺前向模型
        # 让它的预测越来越准 → 习惯化 → 内在奖赏降低 → 探索新区域
        # ==========================================
        forward_loss = self.dopamine.update(state, action, next_state)

        return loss.item(), {
            'r_int': r_int,
            'r_total': total_reward,
            'forward_loss': forward_loss,
        }

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def record_episode(self, ext_reward, int_reward, total_reward):
        """记录每轮的奖赏统计数据"""
        self.episode_ext_rewards.append(ext_reward)
        self.episode_int_rewards.append(int_reward)
        self.episode_rewards.append(total_reward)

    def save(self, path):
        """保存完整模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_ext_rewards': self.episode_ext_rewards,
            'episode_int_rewards': self.episode_int_rewards,
        }, path)
        # 同时保存多巴胺模块
        self.dopamine.save(path.replace('.pth', '_dopamine.pth'))

    def load(self, path):
        """加载完整模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_ext_rewards = checkpoint['episode_ext_rewards']
        self.episode_int_rewards = checkpoint['episode_int_rewards']
        # 加载多巴胺模块
        self.dopamine.load(path.replace('.pth', '_dopamine.pth'))


# ==================== 测试代码 ====================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'g:/code/rengongzhinengdaolun/naoyurenzhi')
    from environment import GridWorld

    print("=== 快速测试：DopAgent 与环境交互 ===\n")

    env = GridWorld(size=8, max_steps=30, seed=42)
    agent = DopAgent(state_dim=2, action_dim=4, beta=0.5, device='cpu')

    state = env.reset()
    total_ext = 0
    total_int = 0

    for step in range(30):
        action = agent.select_action(state)
        next_state, ext_reward, done, info = env.step(action)
        result = agent.update(state, action, ext_reward, next_state, done)

        total_ext += ext_reward

        if result is not None:
            loss, details = result
            total_int += details['r_int']
            print(f"步{step:2d}: 动作={action}, 外奖={ext_reward:.2f}, "
                  f"内奖={details['r_int']:.4f}, "
                  f"总奖={details['r_total']:.4f}, Loss={loss:.4f}")
        else:
            r_int = agent.dopamine.compute_intrinsic_reward(
                state, action, next_state)
            total_int += r_int
            print(f"步{step:2d}: 动作={action}, 外奖={ext_reward:.2f}, "
                  f"内奖={r_int:.4f}, (经验收集中...)")

        state = next_state
        if done:
            print(f"\n{'🎉 到达终点！' if env.agent_pos == env.goal_pos else '⏰ 超时'}")
            break

    print(f"\n外部总奖赏: {total_ext:.2f}")
    print(f"内在总奖赏: {total_int:.4f}")
    print("DopAgent 核心功能正常！🎯")
