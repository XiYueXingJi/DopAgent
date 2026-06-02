"""
训练脚本: 基线 DQN (对照组) vs DopAgent (DQN + 多巴胺内在奖赏)
"""

import sys, os, numpy as np, torch
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from environment import GridWorld
from dqn_agent import DQNAgent
from dop_agent import DopAgent


def create_env(seed=None):
    return GridWorld(size=GRID_SIZE, max_steps=MAX_STEPS,
                     wall_prob=WALL_PROB, start_pos=START_POS,
                     goal_pos=GOAL_POS, seed=seed,
                     action_noise=ACTION_NOISE, use_onehot=False)


def train_baseline(episodes=EPISODES, verbose=True):
    print("=" * 60)
    print("Baseline DQN")
    print("=" * 60)
    env = create_env(seed=42)
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM,
                     lr=LR, gamma=GAMMA, epsilon_start=EPSILON_START,
                     epsilon_end=0.01, epsilon_decay=0.950,  # 极快衰减,约70轮停止
                     memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE,
                     target_update=TARGET_UPDATE, device=DEVICE)
    ep_rewards, ep_success = [], []
    sw = deque(maxlen=100)
    traj_data = []
    for ep in range(1, episodes + 1):
        s = env.reset(); ep_r = 0
        for _ in range(MAX_STEPS):
            a = agent.select_action(s)
            ns, r, d, _ = env.step(a)
            agent.update(s, a, r, ns, d); ep_r += r; s = ns
            if d: break
        ok = env.agent_pos == env.goal_pos
        sw.append(ok); ep_rewards.append(ep_r); ep_success.append(ok)
        agent.record_episode(ep_r); agent.decay_epsilon()
        if ep % EVAL_INTERVAL == 0:
            es, et = evaluate(agent, env, EVAL_EPISODES)
            traj_data.append({'episode': ep, 'success_rate': es/EVAL_EPISODES, 'trajectories': et})
            if verbose: print(f"  Ep {ep:4d}/{episodes} | 奖励: {ep_r:6.2f} | e: {agent.epsilon:.3f} | 近100: {sum(sw)/len(sw):.1%} | 评估: {es}/{EVAL_EPISODES}")
    agent.save(os.path.join(MODELS_DIR, "baseline_dqn.pth"))
    r = {'episode_rewards': ep_rewards, 'episode_success': ep_success, 'trajectory_data': traj_data}
    torch.save(r, os.path.join(RESULTS_DIR, "baseline_results.pt"))
    if verbose: print(f"\nBaseline done: last 100 = {sum(ep_success[-100:])}/100\n")
    return ep_rewards, ep_success, traj_data


def train_dopagent(episodes=EPISODES, verbose=True):
    print("\n" + "=" * 60)
    print("DopAgent (DQN + Dopamine)")
    print("=" * 60)
    env = create_env(seed=42)
    agent = DopAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM,
                     beta=BETA, lr=LR, gamma=GAMMA,
                     epsilon_start=EPSILON_START, epsilon_end=0.05,
                     epsilon_decay=0.993,
                     memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE,
                     target_update=TARGET_UPDATE,
                     forward_lr=FORWARD_LR, forward_hidden=FORWARD_HIDDEN,
                     device=DEVICE)
    ep_ext, ep_int, ep_total, ep_ok = [], [], [], []
    sw = deque(maxlen=100)
    traj_data = []
    for ep in range(1, episodes + 1):
        s = env.reset(); eext=0; eint=0; etot=0
        for _ in range(MAX_STEPS):
            a = agent.select_action(s)
            ns, er, d, _ = env.step(a)
            result = agent.update(s, a, er, ns, d)
            eext += er
            if result is not None:
                _, info = result; eint += info['r_int']; etot += info['r_total']
            else:
                ri = agent.dopamine.compute_intrinsic_reward(s, a, ns)
                eint += ri; etot += er + BETA*ri
            s = ns
            if d: break
        ok = env.agent_pos == env.goal_pos
        sw.append(ok); ep_ext.append(eext); ep_int.append(eint)
        ep_total.append(etot); ep_ok.append(ok)
        agent.record_episode(eext, eint, etot); agent.decay_epsilon()
        if ep % EVAL_INTERVAL == 0:
            es, et = evaluate_dop(agent, env, EVAL_EPISODES)
            traj_data.append({'episode': ep, 'success_rate': es/EVAL_EPISODES, 'trajectories': et})
            if verbose: print(f"  Ep {ep:4d}/{episodes} | 外奖: {eext:6.2f} | 内奖: {eint:6.2f} | 近100: {sum(sw)/len(sw):.1%} | 评估: {es}/{EVAL_EPISODES}")
    agent.save(os.path.join(MODELS_DIR, "dopagent.pth"))
    r = {'episode_rewards': ep_total, 'episode_ext_rewards': ep_ext,
         'episode_int_rewards': ep_int, 'episode_success': ep_ok,
         'trajectory_data': traj_data}
    torch.save(r, os.path.join(RESULTS_DIR, "dopagent_results.pt"))
    if verbose: print(f"\nDopAgent done: last 100 = {sum(ep_ok[-100:])}/100\n")
    return ep_total, ep_ext, ep_int, ep_ok, traj_data


def evaluate(agent, env, n=10):
    ok=0; trajs=[]
    for _ in range(n):
        s=env.reset(); t=[env.agent_pos]
        for _ in range(MAX_STEPS):
            a=agent.select_action(s, eval_mode=True)
            ns,_,d,_=env.step(a); t.append(env.agent_pos); s=ns
            if d: break
        if env.agent_pos==env.goal_pos: ok+=1
        trajs.append(t)
    return ok, trajs


def evaluate_dop(agent, env, n=10):
    ok=0; trajs=[]
    for _ in range(n):
        s=env.reset(); t=[env.agent_pos]
        for _ in range(MAX_STEPS):
            a=agent.select_action(s, eval_mode=True)
            ns,_,d,_=env.step(a); t.append(env.agent_pos); s=ns
            if d: break
        if env.agent_pos==env.goal_pos: ok+=1
        trajs.append(t)
    return ok, trajs
