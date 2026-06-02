"""
=============================================================================
脑与认知科学基础 - 课程项目
多巴胺驱动的好奇心强化学习 (Dopamine-Driven Curiosity RL)

全局配置文件
=============================================================================
"""

import torch
import os

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 确保目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ==================== 设备配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 环境配置 ====================
GRID_SIZE = 12               # 迷宫大小 (12x12)
MAX_STEPS = GRID_SIZE * 4    # 每轮最大步数
WALL_PROB = 0.15             # 随机生成墙壁的概率
START_POS = (0, 0)           # 起点坐标
GOAL_POS = (GRID_SIZE - 1, GRID_SIZE - 1)  # 终点坐标

# 状态是智能体的 (x, y) 坐标，归一化到 [0, 1]
STATE_DIM = 2
# 动作: 0=上, 1=右, 2=下, 3=左
ACTION_DIM = 4

# ==================== DQN 超参数 ====================
LR = 1e-3                    # Q网络学习率
GAMMA = 0.99                 # 折扣因子
BATCH_SIZE = 64              # 批量大小
MEMORY_SIZE = 10000          # 经验回放池大小
TARGET_UPDATE = 100          # 目标网络更新频率（步数）
MIN_MEMORY = 500             # 开始训练的最小经验数

# Epsilon-greedy 探索策略
EPSILON_START = 1.0          # 初始探索率
EPSILON_END = 0.05           # 最终探索率
EPSILON_DECAY = 0.995        # 每轮衰减系数

# ==================== 多巴胺模块超参数 ====================
BETA = 0.5                   # 内在奖赏系数（好奇心权重）
FORWARD_LR = 1e-3            # 前向预测网络学习率
FORWARD_HIDDEN = [128, 64]   # 前向网络隐藏层

# ==================== 训练配置 ====================
EPISODES = 800               # 训练总轮数
EVAL_INTERVAL = 50           # 每多少轮评估一次
EVAL_EPISODES = 10           # 评估时的轮数
RENDER_EVAL = False          # 评估时是否渲染

# ==================== 可视化配置 ====================
FIGURE_DPI = 150
COLORS = {
    "baseline": "#E74C3C",   # 红色 - 基线DQN（对照组，预期失败）
    "dopagent": "#2ECC71",   # 绿色 - DopAgent（核心算法，预期成功）
}
