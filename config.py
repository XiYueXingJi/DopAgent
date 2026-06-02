"""
=============================================================================
全局配置 —— 多巴胺驱动的好奇心强化学习 (最终版)
=============================================================================
"""

import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 环境 ====================
GRID_SIZE = 8                # 8x8 网格
MAX_STEPS = GRID_SIZE * 3    # 24 步
WALL_PROB = 0.15
ACTION_NOISE = 0.15          # 动作噪声让前向模型无法完美预测
START_POS = (0, 0)
GOAL_POS = (GRID_SIZE - 1, GRID_SIZE - 1)
STATE_DIM = 2                # 归一化坐标
ACTION_DIM = 4

# ==================== DQN ====================
LR = 1e-3
GAMMA = 0.99
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 100
EPSILON_START = 1.0

# ==================== 多巴胺模块 ====================
BETA = 5.0                   # 高好奇心
FORWARD_LR = 1e-3
FORWARD_HIDDEN = [128, 64]

# ==================== 训练 ====================
EPISODES = 800
EVAL_INTERVAL = 50
EVAL_EPISODES = 10

FIGURE_DPI = 150
COLORS = {"baseline": "#E74C3C", "dopagent": "#2ECC71"}
