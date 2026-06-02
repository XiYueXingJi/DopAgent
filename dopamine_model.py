"""
=============================================================================
多巴胺前向预测网络 (Dopamine Forward Model)

这是成员C的核心创新模块 —— "类脑"算法的数学实现！

认知科学理论基础：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Wolfram Schultz (1997) 发现：中脑多巴胺神经元编码"奖励预测误差 (RPE)"
- 当实际奖励 > 预期奖励 → 多巴胺神经元时相性兴奋（正向RPE）→ "学到了！"
- 当实际奖励 = 预期奖励 → 多巴胺神经元基线发放 → "习惯了，没有新信息"
- 当实际奖励 < 预期奖励 → 多巴胺神经元抑制（负向RPE）→ "比预期差"

本项目将 RPE 理论推广到"状态预测"：
- 智能体预测 "执行动作A后，环境会变成什么样子"
- 预测误差 (MSE) = 内在奖赏 → 误差越大 = 越新奇 = 越值得探索！
- 随着预测越来越准 → 内在奖赏降低 → 习惯化 (Habituation) → 转向新区域
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

代码量不大，但这是整个项目的"理论 → 代码"关键转化！
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DopamineForwardModel(nn.Module):
    """
    多巴胺前向预测网络

    功能：输入 (当前状态 + 执行的动作) → 预测下一个状态
    本质：这是一个简单的 MLP，但它模拟了基底神经节-皮层回路中的
          "前向模型 (Forward Model)"——大脑用来预测行为后果的机制

    架构: (state_dim + action_dim) → 128 → 64 → state_dim
    """

    def __init__(self, state_dim, action_dim, hidden_dims=[128, 64]):
        """
        参数:
            state_dim: 状态维度（本项目 = 2: 归一化的 x, y 坐标）
            action_dim: 动作维度（本项目 = 4: 上右下左）
            hidden_dims: 隐藏层维度列表
        """
        super().__init__()

        layers = []
        input_dim = state_dim + action_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim

        # 最终输出层：预测下一个状态
        layers.append(nn.Linear(input_dim, state_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state, action_onehot):
        """
        前向传播：预测下一个状态

        参数:
            state: [batch_size, state_dim] 当前状态
            action_onehot: [batch_size, action_dim] 动作的 one-hot 编码

        返回:
            predicted_next_state: [batch_size, state_dim] 预测的下一状态
        """
        # 拼接状态和动作 → 输入前向模型
        x = torch.cat([state, action_onehot], dim=-1)
        predicted_next_state = self.network(x)
        return predicted_next_state


class DopamineModule:
    """
    多巴胺模块（封装前向模型的训练和内在奖赏计算）

    这是项目中最关键的类 —— 它把抽象的脑科学理论变成了可计算的代码！
    """

    def __init__(self, state_dim, action_dim, lr=1e-3, hidden_dims=[128, 64],
                 device='cpu'):
        """
        参数:
            state_dim, action_dim: 状态和动作维度
            lr: 前向模型的学习率
            hidden_dims: 隐藏层大小
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # 初始化前向预测网络
        self.model = DopamineForwardModel(
            state_dim, action_dim, hidden_dims
        ).to(device)

        # 优化器：用于更新前向模型（让它越来越准确地预测）
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # MSE损失函数就是我们的"预测误差量化器"
        self.loss_fn = nn.MSELoss()

        # 记录内在奖赏（用于分析学习过程）
        self.intrinsic_rewards = []

    def compute_intrinsic_reward(self, state, action, next_state):
        """
        计算内在奖赏（好奇心信号）

        核心公式（认知科学 → 数学）:
            R_int = MSE(预测的下一状态, 实际的下一状态)

        直觉理解:
            - 熟悉场景 → 预测准确 (MSE小) → R_int小 → "不好奇了"
            - 新奇场景 → 预测不准 (MSE大) → R_int大 → "多巴胺发放！去探索！"

        参数:
            state: 当前状态 [state_dim]
            action: 执行的动作 (整数, 0-3)
            next_state: 实际的下一状态 [state_dim]

        返回:
            intrinsic_reward: 内在奖赏值 (标量)
        """
        # 转为张量
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        # 动作的 one-hot 编码
        action_onehot = torch.zeros(1, self.action_dim).to(self.device)
        action_onehot[0, action] = 1.0

        # 前向模型预测
        with torch.no_grad():
            predicted_next = self.model(state_t, action_onehot)

        # 预测误差 (MSE) = 内在奖赏
        # 误差越大 → 越新奇 → 内在奖赏越高！
        intrinsic_reward = F.mse_loss(predicted_next, next_state_t, reduction='mean')

        return intrinsic_reward.item()

    def update(self, state, action, next_state):
        """
        更新前向预测网络

        这对应认知科学中的"习惯化"过程：
        智能体反复经历同一场景 → 前向模型越预测越准 → MSE减小 → 内在奖赏降低
        → 智能体被"驱赶"去探索新区域！

        参数:
            state: 当前状态
            action: 执行的动作
            next_state: 实际的下一状态

        返回:
            loss: 预测损失值 (训练用)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        # 动作 one-hot
        action_onehot = torch.zeros(1, self.action_dim).to(self.device)
        action_onehot[0, action] = 1.0

        # 前向传播：预测下一状态
        predicted_next = self.model(state_t, action_onehot)

        # 计算 MSE 损失
        loss = self.loss_fn(predicted_next, next_state_t)

        # 反向传播：更新前向模型参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def record_intrinsic(self, value):
        """记录内在奖赏"""
        self.intrinsic_rewards.append(value)

    def save(self, path):
        """保存模型权重"""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'intrinsic_rewards': self.intrinsic_rewards,
        }, path)

    def load(self, path):
        """加载模型权重"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.intrinsic_rewards = checkpoint['intrinsic_rewards']


# ==================== 测试代码 ====================
if __name__ == "__main__":
    import numpy as np

    print("=== 测试：多巴胺前向预测网络 ===\n")

    # 模拟状态 (归一化坐标) 和动作
    dopamine = DopamineModule(state_dim=2, action_dim=4, device='cpu')

    # 模拟一个转移：从 (0.2, 0.3) 执行"向右"到达 (0.3, 0.3)
    state = np.array([0.2, 0.3], dtype=np.float32)
    action = 1  # 右
    next_state = np.array([0.3, 0.3], dtype=np.float32)

    # 初始预测误差应当较大（网络还没训练）
    r_int = dopamine.compute_intrinsic_reward(state, action, next_state)
    print(f"训练前 - 内在奖赏 (预测误差): {r_int:.6f} (较大)")

    # 训练前向模型几次
    for i in range(100):
        loss = dopamine.update(state, action, next_state)
    print(f"训练100次后 - 最终预测损失: {loss:.6f} (变小)")

    # 再次计算内在奖赏
    r_int = dopamine.compute_intrinsic_reward(state, action, next_state)
    print(f"训练后 - 内在奖赏 (预测误差): {r_int:.6f} (显著减小！)")
    print("\n这证明了习惯化机制：熟悉的转移 → 预测变准 → 内在奖赏降低 ✅")
