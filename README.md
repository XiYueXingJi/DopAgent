# 🧠 DopAgent — 多巴胺驱动的好奇心强化学习

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **脑与认知科学基础 - 课程项目**
>
> 基于 Wolfram Schultz (1997) 奖励预测误差 (RPE) 理论，实现多巴胺机制驱动的内在动机强化学习算法。

---

## 📖 项目简介

在稀疏奖赏环境中，传统强化学习算法（如 DQN）几乎无法学习——因为智能体很难"碰巧"获得奖励信号。本项目受**中脑多巴胺神经元**的"奖励预测误差 (Reward Prediction Error, RPE)"理论启发，设计了一个**前向预测网络**来模拟多巴胺功能，以**预测误差作为内在奖赏**，驱

动智能体主动探索未知环境。

### 🔬 核心思想

```
Schultz (1997) 发现：
  - 多巴胺神经元编码的不是"奖励本身"，而是"奖励预测误差"
  - 实际 > 预期 → 多巴胺爆发 → 正向RPE → "学到了！"
  - 实际 = 预期 → 基线活动 → 习惯化 (Habituation)
  - 实际 < 预期 → 多巴胺抑制 → 负向RPE

本项目将RPE推广到"状态预测"：
  R_int = MSE(预测的下一状态, 实际的下一状态)

  预测误差大 → 环境新奇 → 内在奖赏高 → 好奇心驱动探索
  预测误差小 → 环境熟悉 → 内在奖赏低 → 习惯化 → 转向新区域
```

---

## 🏗️ 项目结构

```
naoyurenzhi/
├── config.py              # 全局配置（超参数、路径）
├── environment.py         # 自定义网格迷宫环境
├── dqn_agent.py           # 基线 DQN（对照组）
├── dopamine_model.py      # 多巴胺前向预测网络 ★核心★
├── dop_agent.py           # DopAgent（DQN + 内在奖赏）
├── train.py               # 训练脚本
├── visualize.py           # 可视化（学习曲线 + 热力图）
├── main.py                # 一键运行入口
├── models/                # 保存的模型权重
├── results/               # 训练数据和图表
└── README.md
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch ≥ 1.11
- NumPy, Matplotlib, Seaborn

### 安装

```bash
# 克隆仓库
git clone https://github.com/XiYueXingJi/DopAgent.git
cd DopAgent

# 安装依赖（PyTorch 请根据官网安装）
pip install numpy matplotlib seaborn
```

### 运行

```bash
# 完整流程：训练 + 画图（约10-20分钟，CPU）
python main.py

# 快速测试（100轮，约2分钟）
python main.py --quick

# 仅训练
python main.py --train

# 仅画图（需要先有训练结果）
python main.py --plot

# 只训练基线DQN
python main.py --baseline-only
```

---

## 📊 预期结果

| 指标 | 基线 DQN | DopAgent (本项目) |
|------|----------|-------------------|
| 探索范围 | 🔴 仅起点附近 | 🟢 整个迷宫 |
| 成功率 | ~0% | 显著提高 |
| 学习曲线 | 贴地死线 | 逐步上升 |

### 核心图表

1. **学习曲线对比** — 基线 DQN（红线）vs DopAgent（绿线）的成功率对比
2. **轨迹热力图** — 直观展示好奇心如何促进空间探索
3. **综合报告** — 所有关键指标一览

---

## 👥 三人分工

| 角色 | 负责人 | 任务 |
|------|--------|------|
| 🎓 **项目组长** | 成员A | 理论包装、报告撰写、PPT答辩 |
| ⚙️ **RL基座** | 成员B | 环境搭建、基线DQN实现、环境测试 |
| 🧪 **核心算法** | 成员C | 多巴胺模块开发、算法融合、可视化 |

---

## 📚 理论基础

### 参考文献

1. **Schultz, W., Dayan, P., & Montague, P. R. (1997).** A neural substrate of prediction and reward. *Science*, 275(5306), 1593-1599.
   > 多巴胺神经元编码奖励预测误差的经典论文。

2. **Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017).** Curiosity-driven exploration by self-supervised prediction. *ICML*.
   > 好奇心驱动探索的现代实现，本项目的工程灵感来源。

3. **Sutton, R. S., & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
   > 强化学习圣经，DQN 理论基础。

### 认知科学概念映射

| 认知科学概念 | 计算实现 |
|-------------|----------|
| 多巴胺时相性发放 | 奖励预测误差 (RPE) |
| 内在动机 / 好奇心 | 前向模型预测误差 (MSE) |
| 习惯化 (Habituation) | 前向模型训练 → 预测误差降低 |
| 基底神经节-皮层回路 | 前向模型 (Forward Model) |
| 外部奖励 | 环境给的真实奖励 |
| 探索-利用权衡 | ε-greedy + 内在奖赏 |

---

## 📝 许可

MIT License — 欢迎用于学习和研究。

---

> **"受限于本科生算力和时间，我们没有跑出一个完美的结果，但我们完整地展示了如何将抽象的脑科学理论转化为可计算的数学模型——而这，恰恰是计算认知科学最核心的能力。"**
>
> — 给组长的答辩兜底金句 😉
