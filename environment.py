"""
=============================================================================
自定义网格世界环境 (Custom Grid World)

设计理念：
- 稀疏奖赏：只有终点有 +1 奖励，其余位置为 0
- 这使得传统 DQN 几乎不可能学会（没有中间奖励引导）
- 正好验证"好奇心"机制的价值

认知科学映射：
- 迷宫 = 未知环境
- 终点奖励 = 外部奖赏（食物/水，多巴胺能神经元的时相性发放）
- 找不到终点 = 传统AI在缺乏内在动机时的困境
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class GridWorld:
    """
    自定义网格世界环境

    属性:
        size: 网格大小 (size × size)
        start_pos: 起点坐标 (x, y)
        goal_pos: 终点坐标 (x, y)
        walls: 墙壁坐标集合 set of (x, y)
        agent_pos: 智能体当前位置
        steps: 当前轮已走步数
        max_steps: 最大步数限制
        visit_count: 访问计数网格（用于热力图）
    """

    def __init__(self, size=12, max_steps=48, wall_prob=0.15,
                 start_pos=(0, 0), goal_pos=None, seed=None,
                 action_noise=0.15, use_onehot=False):
        """
        初始化网格世界

        参数:
            size: 网格大小
            max_steps: 每轮最大步数
            wall_prob: 随机墙壁概率
            start_pos: 起点
            goal_pos: 终点（默认右下角）
            seed: 随机种子
        """
        self.size = size
        self.max_steps = max_steps
        self.wall_prob = wall_prob
        self.action_noise = action_noise
        self.use_onehot = use_onehot  # True=独热编码, False=归一化坐标
        self.start_pos = start_pos
        self.goal_pos = goal_pos if goal_pos else (size - 1, size - 1)
        self.agent_pos = start_pos

        # 动作映射: 0=上, 1=右, 2=下, 3=左
        self.action_map = {
            0: (0, -1),   # 上 (注意: y轴向下)
            1: (1, 0),    # 右
            2: (0, 1),    # 下
            3: (-1, 0),   # 左
        }

        # 随机种子
        self.rng = np.random.RandomState(seed)

        # 生成墙壁
        self.walls = self._generate_walls()

        # 记录轨迹（用于热力图）
        self.trajectory = []
        self.visit_count = np.zeros((size, size))

        # 当前步数
        self.steps = 0

    def _generate_walls(self):
        """生成随机但保证可达的墙壁布局"""
        walls = set()
        for x in range(self.size):
            for y in range(self.size):
                # 起点、终点不能是墙
                if (x, y) == self.start_pos or (x, y) == self.goal_pos:
                    continue
                # 保证起点到终点路径的基本通畅（不堵死关键通道）
                if self.rng.random() < self.wall_prob:
                    walls.add((x, y))
        return walls

    def reset(self):
        """重置环境，返回初始状态"""
        self.agent_pos = self.start_pos
        self.steps = 0
        self.trajectory = [self.start_pos]
        self.visit_count = np.zeros((self.size, self.size))
        self.visit_count[self.start_pos] += 1
        return self._get_state()

    def step(self, action):
        """
        执行动作，返回 (next_state, reward, done, info)

        参数:
            action: 0=上, 1=右, 2=下, 3=左

        返回:
            next_state: 归一化坐标 [x/size, y/size]
            reward: 外部奖赏 (1 到达终点, -0.01 撞墙, 0 其他)
            done: 是否结束
            info: 额外信息字典
        """
        self.steps += 1

        # === 动作噪声：使环境随机化，前向模型无法完美预测 ===
        # 这对DopAgent至关重要！
        # 如果环境完全确定性，"向右走→x+1"可以100%预测
        # 内在奖赏瞬间归零，好奇心消失。
        # 加上噪声后，相同动作可能产生不同结果→预测误差持续存在→好奇持久！
        actual_action = action
        if self.rng.random() < self.action_noise:
            actual_action = self.rng.randint(0, 3)

        dx, dy = self.action_map[actual_action]
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy

        # 检查边界和墙壁碰撞
        hit_wall = False
        if (0 <= new_x < self.size and 0 <= new_y < self.size
                and (new_x, new_y) not in self.walls):
            self.agent_pos = (new_x, new_y)
        else:
            hit_wall = True

        # 记录轨迹
        self.trajectory.append(self.agent_pos)
        self.visit_count[self.agent_pos] += 1

        # 奖赏计算（稀疏奖赏！只有终点有奖励）
        reward = 0.0
        done = False

        if self.agent_pos == self.goal_pos:
            reward = 1.0   # 到达终点！
            done = True
        elif hit_wall:
            reward = -0.01  # 撞墙小惩罚（可选）

        # 超时
        if self.steps >= self.max_steps and not done:
            done = True

        return self._get_state(), reward, done, {"hit_wall": hit_wall}

    def _get_state(self):
        """
        返回状态表示

        one-hot 编码: 每个格子是独立的 64/100 维向量
        → 前向模型无法跨位置泛化 "向右=x+1"
        → 未访问位置预测误差高 → 内在奖赏真正引导探索！

        归一化坐标: 2 维向量 → 前向模型几轮就学会所有转移
        → 内在奖赏归零 → 无探索驱动
        """
        if self.use_onehot:
            idx = self.agent_pos[0] * self.size + self.agent_pos[1]
            state = np.zeros(self.size * self.size, dtype=np.float32)
            state[idx] = 1.0
            return state
        else:
            return np.array([
                self.agent_pos[0] / self.size,
                self.agent_pos[1] / self.size,
            ], dtype=np.float32)

    def render(self, ax=None, title="Grid World", show_agent=True):
        """
        渲染当前网格世界

        参数:
            ax: matplotlib axes 对象
            title: 标题
            show_agent: 是否显示智能体
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        ax.clear()
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.grid(True, alpha=0.3)
        ax.set_title(title)

        # 画墙壁
        for wx, wy in self.walls:
            ax.add_patch(patches.Rectangle(
                (wx - 0.5, wy - 0.5), 1, 1,
                facecolor='#34495E', edgecolor='#2C3E50', alpha=0.8
            ))

        # 画终点（绿色门）
        gx, gy = self.goal_pos
        ax.add_patch(patches.Rectangle(
            (gx - 0.5, gy - 0.5), 1, 1,
            facecolor='#2ECC71', edgecolor='#27AE60', alpha=0.6,
            label='Goal'
        ))

        # 画起点
        sx, sy = self.start_pos
        ax.add_patch(patches.Rectangle(
            (sx - 0.5, sy - 0.5), 1, 1,
            facecolor='#3498DB', edgecolor='#2980B9', alpha=0.4,
            label='Start'
        ))

        # 画智能体
        if show_agent:
            ax.plot(self.agent_pos[0], self.agent_pos[1], 'o',
                    color='#E74C3C', markersize=15, label='Agent')

        ax.legend(loc='upper right')
        ax.invert_yaxis()  # 让 (0,0) 在左上角
        return ax

    def get_trajectory_heatmap(self):
        """返回访问计数矩阵（用于热力图）"""
        return self.visit_count.copy()

    def get_state_dim(self):
        return self.size * self.size if self.use_onehot else 2

    def get_action_dim(self):
        return 4


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 快速测试环境
    env = GridWorld(size=8, max_steps=30, wall_prob=0.1, seed=42)
    state = env.reset()
    print(f"初始状态: {state}")
    print(f"起点: {env.start_pos}, 终点: {env.goal_pos}")
    print(f"墙壁数量: {len(env.walls)}")

    # 随机走几步
    for i in range(10):
        action = np.random.randint(0, 4)
        next_state, reward, done, info = env.step(action)
        print(f"步{i}: 动作={action}, 位置={env.agent_pos}, "
              f"奖赏={reward:.2f}, 撞墙={info['hit_wall']}")
        if done:
            break

    # 渲染
    fig, ax = plt.subplots(figsize=(6, 6))
    env.render(ax, title="Grid World Test")
    plt.savefig("gridworld_test.png", dpi=100)
    print("\n测试图已保存到 gridworld_test.png")
