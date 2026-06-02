"""
=============================================================================
🧠 脑与认知科学基础 - 课程项目
多巴胺驱动的好奇心强化学习 (Dopamine-Driven Curiosity RL)

一键运行入口 —— 训练 → 评估 → 可视化
=============================================================================

使用方法:
    python main.py              # 完整流程: 训练 + 画图
    python main.py --train      # 仅训练
    python main.py --plot       # 仅画图（需要先有训练结果）
    python main.py --quick      # 快速测试（少量轮数，验证代码能跑）

三人分工对应:
    成员A (组长):   看图表、写报告、做PPT
    成员B (基座):   运行 baseline 部分，理解 DQN 原理
    成员C (核心):   运行 DopAgent 部分，理解多巴胺模块
"""

import argparse
import sys
import os

# 确保能导入同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *


def main():
    parser = argparse.ArgumentParser(
        description='🧠 多巴胺好奇心强化学习 - 脑与认知科学项目'
    )
    parser.add_argument('--train', action='store_true',
                        help='运行训练（基线DQN + DopAgent）')
    parser.add_argument('--plot', action='store_true',
                        help='生成可视化图表（需要已训练）')
    parser.add_argument('--quick', action='store_true',
                        help='快速测试模式（减少训练轮数）')
    parser.add_argument('--baseline-only', action='store_true',
                        help='只训练基线DQN')

    args = parser.parse_args()

    # 如果没有任何参数，默认运行完整流程
    run_all = not (args.train or args.plot or args.quick or args.baseline_only)

    print("""
╔══════════════════════════════════════════════════════════════╗
║          🧠 脑与认知科学基础 - 课程项目                      ║
║    多巴胺驱动的好奇心强化学习 (DopAgent)                      ║
║                                                              ║
║    基于 Schultz (1997) RPE 理论                               ║
║    "多巴胺神经元编码奖励预测误差"                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    print(f"📍 设备: {DEVICE}")
    print(f"📍 迷宫: {GRID_SIZE}×{GRID_SIZE}, 墙壁概率: {WALL_PROB}")
    print(f"📍 训练轮数: {'100 (快速测试)' if args.quick else EPISODES}")
    print(f"📍 多巴胺系数 β: {BETA}\n")

    # ---- 训练阶段 ----
    if run_all or args.train or args.quick or args.baseline_only:
        from train import train_baseline, train_dopagent

        episodes = 100 if args.quick else EPISODES

        print("=" * 60)
        print("PHASE 1: 训练基线 DQN (对照组)")
        print("=" * 60)
        bl_results = train_baseline(episodes=episodes)

        if not args.baseline_only:
            print("\n" + "=" * 60)
            print("PHASE 2: 训练 DopAgent (核心创新)")
            print("=" * 60)
            dop_results = train_dopagent(episodes=episodes)

    # ---- 可视化阶段 ----
    if run_all or args.plot:
        from visualize import plot_all
        print("\n" + "=" * 60)
        print("PHASE 3: 生成可视化图表")
        print("=" * 60)
        plot_all()

    print("""
╔══════════════════════════════════════════════════════════════╗
║                    ✅ 全部完成！                              ║
║                                                              ║
║  查看结果: results/ 文件夹                                    ║
║    - baseline_results.pt  (基线DQN训练数据)                   ║
║    - dopagent_results.pt  (DopAgent训练数据)                  ║
║    - 1_learning_curves.png  (学习曲线)                        ║
║    - 2_trajectory_heatmaps.png (热力图)                       ║
║    - 3_summary_report.png  (综合报告)                         ║
║                                                              ║
║  模型: models/ 文件夹                                         ║
║    - baseline_dqn.pth  (基线DQN模型)                          ║
║    - dopagent.pth      (DopAgent模型)                         ║
║                                                              ║
║  下一步:                                                      ║
║    - 成员A: 用这些图写报告和PPT                                ║
║    - 成员B: 分析基线DQN为什么失败                              ║
║    - 成员C: 分析内在奖赏如何驱动探索                            ║
╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
