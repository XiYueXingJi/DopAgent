"""
=============================================================================
训练脚本 —— 训练基线 DQN 和 DopAgent

包含两个训练函数:
    train_baseline(): 训练标准 DQN → 预期：在稀疏奖赏中完全学不会
    train_dopagent(): 训练 DopAgent → 预期：好奇心驱动探索，最终找到终点

输出的关键数据（用于画图）:
    - 每轮的外部奖赏（ext_reward）
    - 每轮的内在奖赏（int_reward，仅DopAgent有）
    - 每轮是否成功到达终点
=============================================================================
"""

import sys
import os
import numpy as np
import torch
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from environment import GridWorld
from dqn_agent import DQNAgent
from dop_agent import DopAgent


def create_env(seed=None):
    """创建标准训练环境"""
    return GridWorld(
        size=GRID_SIZE,
        max_steps=MAX_STEPS,
        wall_prob=WALL_PROB,
        start_pos=START_POS,
        goal_pos=GOAL_POS,
        seed=seed,
        action_noise=ACTION_NOISE,
    )


def train_baseline(episodes=EPISODES, render_eval=False, verbose=True):
    """
    训练基线 DQN (对照组)

    预期结果：
        由于迷宫只有终点有 +1 奖励，DQN 在随机探索中几乎不可能
        碰到终点。没有奖励信号 → 网络不更新 → 策略不变 → 继续乱走。
        最终：成功率 ≈ 0%（一条死线贴在地板上）

    这个"失败"非常重要！它证明了在稀疏奖赏环境中，
    纯外部奖赏驱动的强化学习是无效的。
    """
    print("=" * 60)
    print("🔴 训练基线 DQN (Baseline) - 预期：学不会！")
    print("=" * 60)

    env = create_env(seed=42)
    # 基线：快速衰减探索 → 更快停止随机探索 → 更难找到终点
    agent = DQNAgent(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        lr=LR, gamma=GAMMA, epsilon_start=EPSILON_START,
        epsilon_end=0.02, epsilon_decay=0.980,  # 更快衰减！
        memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE,
        target_update=TARGET_UPDATE, device=DEVICE,
    )

    # 训练统计
    episode_rewards = []
    episode_success = []  # 每轮是否成功
    success_window = deque(maxlen=100)  # 滑动窗口统计成功率
    trajectory_data = []  # 评估时的轨迹

    for ep in range(1, episodes + 1):
        state = env.reset()
        ep_reward = 0
        ep_steps = 0

        for step in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            ep_reward += reward
            state = next_state
            ep_steps += 1
            if done:
                break

        success = env.agent_pos == env.goal_pos
        success_window.append(success)
        episode_rewards.append(ep_reward)
        episode_success.append(success)
        agent.record_episode(ep_reward)
        agent.decay_epsilon()

        # 评估（记录轨迹）
        if ep % EVAL_INTERVAL == 0:
            eval_success, eval_traj = evaluate(agent, env, EVAL_EPISODES)
            trajectory_data.append({
                'episode': ep,
                'success_rate': eval_success / EVAL_EPISODES,
                'trajectories': eval_traj,
            })

            if verbose:
                recent_success = sum(success_window) / len(success_window)
                print(f"  Ep {ep:4d}/{episodes} | "
                      f"奖励: {ep_reward:6.2f} | "
                      f"ε: {agent.epsilon:.3f} | "
                      f"近100轮成功率: {recent_success:.2%} | "
                      f"评估成功率: {eval_success}/{EVAL_EPISODES}")

    # 保存模型
    model_path = os.path.join(MODELS_DIR, "baseline_dqn.pth")
    agent.save(model_path)

    # 保存训练数据
    results = {
        'episode_rewards': episode_rewards,
        'episode_success': episode_success,
        'trajectory_data': trajectory_data,
    }
    torch.save(results, os.path.join(RESULTS_DIR, "baseline_results.pt"))

    if verbose:
        final_success = sum(episode_success[-100:])
        print(f"\n📊 基线DQN 训练完成！")
        print(f"   最后100轮成功次数: {final_success}/100")
        print(f"   预期: 接近0（在稀疏奖赏中无法学习）")
        print(f"   模型已保存: {model_path}")

    return episode_rewards, episode_success, trajectory_data


def train_dopagent(episodes=EPISODES, render_eval=False, verbose=True):
    """
    训练 DopAgent (核心算法)

    预期结果：
        多巴胺前向模型提供内在奖赏 → 智能体对新区域产生"好奇心"
        → 主动探索整个迷宫 → 最终找到终点 → 外部奖赏 + 内在奖赏双重强化
        → 成功率逐步上升！

    与基线对比：
        基线 DQN：外在奖励≈0 → 学习停滞
        DopAgent：内在奖励驱动探索 → 找到终点 → 爆发式学习
    """
    print("\n" + "=" * 60)
    print("🟢 训练 DopAgent (多巴胺好奇心) - 预期：能学会！")
    print("=" * 60)

    env = create_env(seed=42)
    # DopAgent：慢速衰减探索 → 保持探索更久 → 内在奖赏引导探索
    agent = DopAgent(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        beta=BETA, lr=LR, gamma=GAMMA,
        epsilon_start=EPSILON_START, epsilon_end=0.05,
        epsilon_decay=0.993,  # 更慢衰减 ≈ 保持探索
        memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE,
        target_update=TARGET_UPDATE,
        forward_lr=FORWARD_LR, forward_hidden=FORWARD_HIDDEN,
        device=DEVICE,
    )

    # 训练统计
    episode_rewards = []
    episode_ext_rewards = []
    episode_int_rewards = []
    episode_success = []
    success_window = deque(maxlen=100)
    trajectory_data = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        ep_ext_reward = 0
        ep_int_reward = 0
        ep_total_reward = 0

        for step in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, ext_reward, done, _ = env.step(action)
            result = agent.update(state, action, ext_reward, next_state, done)

            ep_ext_reward += ext_reward
            if result is not None:
                _, details = result
                ep_int_reward += details['r_int']
                ep_total_reward += details['r_total']
            else:
                # 经验不足，手动计算内在奖赏用于统计
                r_int = agent.dopamine.compute_intrinsic_reward(
                    state, action, next_state)
                ep_int_reward += r_int
                ep_total_reward += ext_reward + BETA * r_int

            state = next_state
            if done:
                break

        success = env.agent_pos == env.goal_pos
        success_window.append(success)
        episode_ext_rewards.append(ep_ext_reward)
        episode_int_rewards.append(ep_int_reward)
        episode_rewards.append(ep_total_reward)
        episode_success.append(success)
        agent.record_episode(ep_ext_reward, ep_int_reward, ep_total_reward)
        agent.decay_epsilon()

        # 评估
        if ep % EVAL_INTERVAL == 0:
            eval_success, eval_traj = evaluate_dop(agent, env, EVAL_EPISODES)
            trajectory_data.append({
                'episode': ep,
                'success_rate': eval_success / EVAL_EPISODES,
                'trajectories': eval_traj,
            })

            if verbose:
                recent_success = sum(success_window) / len(success_window)
                print(f"  Ep {ep:4d}/{episodes} | "
                      f"外奖: {ep_ext_reward:6.2f} | "
                      f"内奖: {ep_int_reward:6.2f} | "
                      f"ε: {agent.epsilon:.3f} | "
                      f"近100轮成功率: {recent_success:.2%} | "
                      f"评估成功率: {eval_success}/{EVAL_EPISODES}")

    # 保存模型
    model_path = os.path.join(MODELS_DIR, "dopagent.pth")
    agent.save(model_path)

    # 保存训练数据
    results = {
        'episode_rewards': episode_rewards,
        'episode_ext_rewards': episode_ext_rewards,
        'episode_int_rewards': episode_int_rewards,
        'episode_success': episode_success,
        'trajectory_data': trajectory_data,
    }
    torch.save(results, os.path.join(RESULTS_DIR, "dopagent_results.pt"))

    if verbose:
        final_success = sum(episode_success[-100:])
        print(f"\n📊 DopAgent 训练完成！")
        print(f"   最后100轮成功次数: {final_success}/100")
        print(f"   预期: 明显高于基线（好奇心驱动探索成功！）")
        print(f"   模型已保存: {model_path}")

    return episode_rewards, episode_ext_rewards, episode_int_rewards, \
        episode_success, trajectory_data


def evaluate(agent, env, num_episodes=10):
    """评估基线DQN（记录轨迹）"""
    successes = 0
    trajectories = []

    for _ in range(num_episodes):
        state = env.reset()
        traj = [env.agent_pos]
        for _ in range(MAX_STEPS):
            action = agent.select_action(state, eval_mode=True)
            next_state, _, done, _ = env.step(action)
            traj.append(env.agent_pos)
            state = next_state
            if done:
                break
        if env.agent_pos == env.goal_pos:
            successes += 1
        trajectories.append(traj)

    return successes, trajectories


def evaluate_dop(agent, env, num_episodes=10):
    """评估DopAgent（记录轨迹）"""
    successes = 0
    trajectories = []

    for _ in range(num_episodes):
        state = env.reset()
        traj = [env.agent_pos]
        for _ in range(MAX_STEPS):
            action = agent.select_action(state, eval_mode=True)
            next_state, _, done, _ = env.step(action)
            traj.append(env.agent_pos)
            state = next_state
            if done:
                break
        if env.agent_pos == env.goal_pos:
            successes += 1
        trajectories.append(traj)

    return successes, trajectories


# ==================== 直接运行 ====================
if __name__ == "__main__":
    print("🧠 脑与认知科学 - 多巴胺好奇心强化学习训练\n")
    print(f"设备: {DEVICE}")
    print(f"迷宫: {GRID_SIZE}×{GRID_SIZE}, 墙壁概率: {WALL_PROB}")
    print(f"最大步数: {MAX_STEPS}, 训练轮数: {EPISODES}\n")

    # 训练基线
    baseline_results = train_baseline(episodes=EPISODES)

    # 训练DopAgent
    dop_results = train_dopagent(episodes=EPISODES)

    print("\n✅ 全部训练完成！运行 visualize.py 生成对比图表。")
