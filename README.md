# 🧠 DopAgent — 多巴胺驱动的好奇心强化学习

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **脑与认知科学基础 - 课程项目**
>
> 基于 Wolfram Schultz (1997) 奖励预测误差 (RPE) 理论，实现多巴胺机制驱动的内在动机强化学习算法。

---

## 📖 项目简介

在稀疏奖赏环境中，传统 DQN 几乎无法学习——智能体很难"碰巧"获得奖励信号。本项目受**中脑多巴胺神经元**的"奖励预测误差 (Reward Prediction Error, RPE)"理论启发，设计了一个**前向预测网络**来模拟多巴胺功能，以**预测误差作为内在奖赏**，驱动智能体主动探索未知环境。

### 🎯 实际结果

| | Baseline DQN | **DopAgent** |
|---|---|---|
| 总成功次数 | 0/800 (0%) | **203/800 (25%)** |
| 最后100轮成功率 | 0% | **43%** |
| 峰值成功率 | 0% | **59%** |
| 评估最好成绩 | 0/10 | **10/10** |
| 内在奖赏趋势 | N/A | 0.22 → 0.05（习惯化！） |

---

## 🔬 核心思想

```
Schultz (1997) 发现：
  - 多巴胺神经元编码的不是"奖励本身"，而是"奖励预测误差"
  - 实际 > 预期 → 多巴胺爆发 → 正向RPE → "学到了！"
  - 实际 = 预期 → 基线活动 → 习惯化 (Habituation)
  - 实际 < 预期 → 多巴胺抑制 → 负向RPE

本项目将RPE推广到"状态预测"：
  R_int = MSE(f_forward(s, a), s')

  预测误差大 → 环境新奇 → 内在奖赏高 → 好奇心驱动探索
  预测误差小 → 环境熟悉 → 内在奖赏低 → 习惯化 → 转向新区域
```

### 关键技术设计

1. **动作噪声**（15%）：智能体说的动作有 15% 概率被随机替换。真实大脑面对的环境充满不确定性，噪声使前向模型无法过拟合，维持持续的好奇心。

2. **动态内在奖赏重算**：经验回放池只存外部奖赏。训练时用当前最新的前向模型**实时重算**内在奖赏。对应认知科学洞察：多巴胺 RPE 计算是"在线"的，大脑不会记住"某件事以前让我惊讶"。

3. **差异化探索策略**：基线 ε 衰减 0.95（~70轮停止探索），DopAgent ε 衰减 0.993（持续探索）。基线模拟"缺乏多巴胺"的状态——快速停止探索后策略锁死。DopAgent 靠内在奖赏在 ε 降低后仍能定向探索。

---

## 🏗️ 项目结构

```
naoyurenzhi/
├── config.py              # 全局配置
├── environment.py         # 自定义网格迷宫（8×8，15%噪声）
├── dqn_agent.py           # 基线 DQN（对照组）
├── dopamine_model.py      # 多巴胺前向预测网络 ★核心创新★
├── dop_agent.py           # DopAgent（DQN + 动态内在奖赏）
├── train.py               # 训练脚本
├── visualize.py           # 可视化（学习曲线 + 热力图 + 综合报告）
├── main.py                # 一键运行
├── results/               # 实验图表 + 说明文档
└── README.md
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch ≥ 1.11
- NumPy, Matplotlib, Seaborn

### 运行

```bash
# 完整流程：训练 + 画图（约10分钟，CPU）
python main.py

# 快速测试（100轮，约1分钟）
python main.py --quick
```

---

## 📊 实验结果

### 核心图表

| 图 | 内容 | 要点 |
|----|------|------|
| `1_learning_curves.png` | 学习曲线对比 | 红线贴死 0%，绿线拔地而起冲到 59% |
| `2_trajectory_heatmaps.png` | 轨迹热力图 | 左：困在起点 → 右：铺满迷宫 |
| `3_summary_report.png` | 综合报告 | 6 个子图 + 核心公式 + 关键数字 |

### 实验配置

| 参数 | 值 |
|------|-----|
| 迷宫 | 8×8，15% 墙壁，15% 动作噪声 |
| β（多巴胺系数） | 5.0 |
| 训练轮数 | 800 |
| 基线 ε 衰减 | 0.95（~70轮后 ε→0.01） |
| DopAgent ε 衰减 | 0.993（保留探索能力） |

详细结果解读见 [results/README.md](results/README.md)。

---

## 👥 三人分工

| 角色 | 任务 |
|------|------|
| 🎓 项目组长 | 理论包装、报告撰写、PPT答辩 |
| ⚙️ RL基座 | 环境搭建、基线DQN实现 |
| 🧪 核心算法 | 多巴胺模块开发、算法融合、可视化 |

---

## 📚 理论基础

### 参考文献

1. **Schultz, W., Dayan, P., & Montague, P. R. (1997).** A neural substrate of prediction and reward. *Science*, 275(5306), 1593-1599.

2. **Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017).** Curiosity-driven exploration by self-supervised prediction. *ICML*.

3. **Sutton, R. S., & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

### 认知科学 → 计算映射

| 认知科学概念 | 计算实现 | 数据证据 |
|-------------|----------|---------|
| 奖励预测误差 (RPE) | R_int = MSE(f(s,a), s') | 内在奖赏持续存在 |
| 习惯化 (Habituation) | 前向模型训练 → MSE降低 | 内奖 0.22→0.05 |
| 内在动机 | β × R_int 驱动探索 | 热力图铺满迷宫 |
| 多巴胺时相性发放 | 高预测误差 → 高内奖 | 成功率曲线上升 |

---

## 📝 许可

MIT License
