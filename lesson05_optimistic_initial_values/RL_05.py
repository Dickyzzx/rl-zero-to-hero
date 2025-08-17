# filename: bandit_optimistic_initial_values.py
import argparse                  # 命令行参数解析
import numpy as np               # 数值计算库（随机数、数组运算等）
import matplotlib.pyplot as plt  # 绘图库
from typing import Tuple, List   # 类型注解（便于阅读和IDE提示）


class TenArmedBandit:
    """
    10-臂老虎机环境类

    输入参数:
    - k (int, 默认=10): 动作臂数量
    - q_mean (float, 默认=0.0): q*(a) 的总体均值
    - q_std (float, 默认=1.0): q*(a) 的总体标准差
    - reward_std (float, 默认=1.0): 即时奖励分布的标准差 (R ~ N(q*(a), reward_std^2))
    - rng (np.random.Generator, 默认=None): 随机数生成器

    输出:
    - 提供 step(action) 方法：给定动作返回一次奖励
    - 提供 reset() 方法：重新为每个动作生成 q*(a) 并记录最优动作

    用途:
    - 模拟经典 10-臂老虎机问题，评估和对比不同动作选择策略
    """

    def __init__(self, k: int = 10, q_mean: float = 0.0, q_std: float = 1.0, reward_std: float = 1.0, rng: np.random.Generator = None):
        self.k = k                                      # 动作臂数量
        self.q_mean = q_mean                            # q*(a) 的总体均值
        self.q_std = q_std                              # q*(a) 的总体标准差
        self.reward_std = reward_std                    # 即时奖励分布的标准差
        self.rng = rng if rng is not None else np.random.default_rng()  # 随机数生成器（若未提供则创建默认）
        self.reset()                                    # 初始化时生成一组 q*(a)

    def reset(self):
        self.q_star = self.rng.normal(loc=self.q_mean, scale=self.q_std, size=self.k)  # 为每个动作生成 q*(a)
        self.optimal_action = int(np.argmax(self.q_star))  # 记录最优动作编号（q*(a) 最大的那个）

    def step(self, action: int) -> float:
        # 输入: action (int) —— 选择的动作臂编号
        # 输出: reward (float) —— 该动作的一次即时奖励采样
        # 作用: 根据 q*(a) 返回一次奖励采样，R_t ~ N(q*(a), reward_std^2)
        return float(self.rng.normal(loc=self.q_star[action], scale=self.reward_std))  # 采样并返回奖励


class OptimisticGreedyAgent:
    """
    支持乐观初值的ε-Greedy策略智能体类

    输入参数:
    - k (int): 动作臂数量
    - epsilon (float): 探索率 ε（以 ε 的概率随机探索）
    - initial_q (float): Q值的初始值（乐观初值的关键参数）
    - alpha (float, 默认=None): 学习率，若为None则使用样本平均法
    - rng (np.random.Generator, 默认=None): 随机数生成器

    输出:
    - select_action() -> int: 按 ε-Greedy 规则选择动作
    - update(action, reward) -> None: 更新 Q 值
    - reset() -> None: 重置 Q 值和计数器

    用途:
    - 通过设置乐观初值来鼓励探索，即使在ε=0的纯贪婪策略下也能进行有效探索
    """

    def __init__(self, k: int, epsilon: float, initial_q: float = 0.0, alpha: float = None, rng: np.random.Generator = None):
        self.k = k                                   # 动作臂数量
        self.epsilon = epsilon                       # 探索率 ε
        self.initial_q = initial_q                   # Q值初始值（乐观初值关键参数）
        self.alpha = alpha                           # 学习率（若为None则使用样本平均）
        self.rng = rng if rng is not None else np.random.default_rng()  # 随机数生成器
        self.reset()                                 # 初始化 Q 和 N

    def reset(self):
        self.Q = np.full(self.k, self.initial_q, dtype=float)  # 用initial_q初始化所有Q值
        self.N = np.zeros(self.k, dtype=int)         # 动作计数 N(a)，初始为 0

    def select_action(self) -> int:
        # 输入: 无
        # 输出: 动作编号 (int)
        # 作用: 以 ε 的概率随机探索，否则选择当前 Q 最大的动作；并对并列最大随机打破
        if self.rng.random() < self.epsilon:         # 以 ε 概率走探索分支
            return int(self.rng.integers(0, self.k)) # 在 [0, k) 内随机选一个动作
        max_val = np.max(self.Q)                     # 找到当前 Q 的最大值
        candidates = np.flatnonzero(self.Q == max_val)  # 在布尔数组中取 True 的索引（即所有并列最大的位置）
        return int(self.rng.choice(candidates))      # 从并列最优候选中随机挑一个，避免总选到下标最小者

    def update(self, action: int, reward: float):
        # 输入: action (int) —— 执行动作的编号；reward (float) —— 刚刚获得的奖励
        # 输出: None
        # 作用: 更新该动作的价值估计 Q(a)
        self.N[action] += 1                          # 该动作计数 +1
        
        if self.alpha is None:
            # 使用样本平均法
            step_size = 1.0 / self.N[action]         # 步长 α_t = 1 / N(a)
        else:
            # 使用固定学习率
            step_size = self.alpha
            
        self.Q[action] += step_size * (reward - self.Q[action])  # 增量更新 Q(a)


def run_single(agent: OptimisticGreedyAgent, bandit: TenArmedBandit, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    单次运行（Single Run）

    输入参数:
    - agent (OptimisticGreedyAgent): 支持乐观初值的智能体
    - bandit (TenArmedBandit): 10-臂老虎机环境
    - steps (int): 运行的时间步数

    输出:
    - rewards (np.ndarray[steps]): 每一步的即时奖励
    - optimal_taken (np.ndarray[steps]): 每一步是否选到最优动作（0/1）

    用途:
    - 执行一次完整的策略交互过程
    """
    rewards = np.zeros(steps, dtype=float)           # 预分配奖励数组
    optimal_taken = np.zeros(steps, dtype=int)       # 预分配是否最优动作的标记数组

    agent.reset()                                    # 重置智能体（重置 Q 与 N）
    bandit.reset()                                   # 重置环境（重新抽取 q*(a)）

    for t in range(steps):                           # 遍历每个时间步
        a = agent.select_action()                    # 选择一个动作
        r = bandit.step(a)                           # 执行动作并获得奖励
        agent.update(a, r)                           # 用奖励更新 Q(a)
        rewards[t] = r                               # 记录该步的奖励
        optimal_taken[t] = 1 if a == bandit.optimal_action else 0  # 记录是否为最优动作

    return rewards, optimal_taken                    # 返回两个序列


def run_optimistic_experiment(
    configs: List[dict],
    runs: int,
    steps: int,
    k: int = 10,
    q_mean: float = 0.0,
    q_std: float = 1.0,
    reward_std: float = 1.0,
    seed: int = 42
):
    """
    乐观初值实验：对比不同初值设置的效果

    输入参数:
    - configs (List[dict]): 配置列表，每个配置包含：
        {"name": "配置名", "epsilon": ε值, "initial_q": Q初值, "alpha": 学习率}
    - runs (int): 独立运行次数
    - steps (int): 每次运行的时间步数
    - k (int): 动作臂数量
    - q_mean (float): q*(a) 总体均值
    - q_std (float): q*(a) 总体标准差
    - reward_std (float): 即时奖励标准差
    - seed (int): 主随机种子

    输出:
    - results (dict): {config_name: (avg_rewards, avg_optimal_pct)}

    用途:
    - 对比不同初值和探索策略的效果，特别是乐观初值与传统ε-贪婪的差异
    """
    rng_master = np.random.default_rng(seed)         # 主随机源
    results = {}                                     # 保存各配置的平均结果

    for config in configs:                           # 遍历每个配置
        config_name = config["name"]
        epsilon = config["epsilon"]
        initial_q = config["initial_q"]
        alpha = config.get("alpha", None)            # 如果没有指定alpha，使用样本平均
        
        print(f"正在运行: {config_name}")
        
        avg_rewards = np.zeros(steps, dtype=float)   # 累加奖励
        avg_optimal = np.zeros(steps, dtype=float)   # 累加是否为最优

        for i in range(runs):                        # 独立运行 runs 次
            run_seed_env = int(rng_master.integers(0, 2**31 - 1))
            run_seed_agent = int(rng_master.integers(0, 2**31 - 1))
            rng_env = np.random.default_rng(run_seed_env)
            rng_agent = np.random.default_rng(run_seed_agent)

            bandit = TenArmedBandit(k=k, q_mean=q_mean, q_std=q_std, reward_std=reward_std, rng=rng_env)
            agent = OptimisticGreedyAgent(k=k, epsilon=epsilon, initial_q=initial_q, alpha=alpha, rng=rng_agent)

            rewards, optimal_taken = run_single(agent, bandit, steps)
            avg_rewards += rewards
            avg_optimal += optimal_taken

        avg_rewards /= runs                          # 取平均奖励
        avg_optimal = (avg_optimal / runs) * 100.0   # 转换为百分比
        results[config_name] = (avg_rewards, avg_optimal)

    return results


def plot_optimistic_curves(results, steps: int, out_prefix: str = "bandit_optimistic"):
    """
    绘制乐观初值实验的对比曲线

    输入参数:
    - results (dict): run_optimistic_experiment 的返回结果
    - steps (int): 时间步数
    - out_prefix (str): 输出文件名前缀

    输出:
    - 保存两张图片：平均奖励曲线和最优动作选择率曲线
    """
    t = np.arange(steps)                             # 横轴坐标

    # 图1：平均奖励对比
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'gray']  # 蓝色表示乐观，灰色表示现实
    linestyles = ['-', '--']   # 实线表示贪婪，虚线表示ε-贪婪
    
    for i, (config_name, (avg_rewards, _)) in enumerate(results.items()):
        color = colors[i % len(colors)]
        style = linestyles[i % len(linestyles)]
        plt.plot(t, avg_rewards, label=config_name, color=color, linestyle=style, linewidth=2)
    
    plt.title("Average Reward: Optimistic vs Realistic Initial Values")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    fname1 = f"{out_prefix}_avg_reward.png"
    plt.tight_layout()
    plt.savefig(fname1, dpi=150, bbox_inches='tight')
    print(f"已保存: {fname1}")

    # 图2：最优动作选择率对比
    plt.figure(figsize=(10, 6))
    
    for i, (config_name, (_, avg_optimal_pct)) in enumerate(results.items()):
        color = colors[i % len(colors)]
        style = linestyles[i % len(linestyles)]
        plt.plot(t, avg_optimal_pct, label=config_name, color=color, linestyle=style, linewidth=2)
    
    plt.title("% Optimal Action: Optimistic vs Realistic Initial Values")
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    fname2 = f"{out_prefix}_optimal_pct.png"
    plt.tight_layout()
    plt.savefig(fname2, dpi=150, bbox_inches='tight')
    print(f"已保存: {fname2}")


def plot_single_optimistic_run(
    config: dict,
    steps: int = 1000,
    k: int = 10,
    seed: int = 123,
    out_prefix: str = "optimistic_single_run"
):
    """
    绘制乐观初值策略的单次运行示例
    """
    rng = np.random.default_rng(seed)
    bandit = TenArmedBandit(k=k, rng=rng)
    agent = OptimisticGreedyAgent(
        k=k,
        epsilon=config["epsilon"],
        initial_q=config["initial_q"],
        alpha=config.get("alpha", None),
        rng=rng
    )
    
    rewards, optimal_taken = run_single(agent, bandit, steps)
    t = np.arange(steps)

    plt.figure(figsize=(10, 6))
    plt.plot(t, rewards, linewidth=1, alpha=0.7)
    plt.title(f"Single Run Example: {config['name']}")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.grid(True, linestyle="--", linewidth=0.5)
    fname = f"{out_prefix}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    print(f"已保存: {fname}")


def main():
    """
    主程序：对比乐观初值与现实初值在多臂赌博机中的表现

    核心对比:
    1. Realistic, ε-greedy: Q₁ = 0, ε = 0.1, α = 0.1
       - 传统方法：初值为0，通过ε-贪婪进行探索
    
    2. Optimistic, greedy: Q₁ = 5.0, ε = 0, α = 0.1  
       - 乐观初值：高初值鼓励探索，即使纯贪婪也会探索

    关键洞察:
    - 乐观初值能让纯贪婪策略也具备探索能力
    - 初期探索更激进，但随着学习收敛到真实值
    - 长期性能往往优于传统ε-贪婪方法
    """
    print("=== 第五课：乐观初值在多臂赌博机中的作用 ===")
    print("对比以下两种策略：")
    print("1. Realistic, ε-greedy: Q₁ = 0, ε = 0.1, α = 0.1")
    print("   - 传统方法：初值为0，通过ε-贪婪进行探索")
    print("2. Optimistic, greedy: Q₁ = 5.0, ε = 0, α = 0.1")
    print("   - 乐观初值：高初值鼓励探索，即使纯贪婪也会探索")
    print()

    # 实验配置
    configs = [
        {
            "name": "Optimistic, greedy\nQ₁ = 5.0, ε = 0, α = 0.1",
            "epsilon": 0.0,
            "initial_q": 5.0,
            "alpha": 0.1
        },
        {
            "name": "Realistic, ε-greedy\nQ₁ = 0, ε = 0.1, α = 0.1", 
            "epsilon": 0.1,
            "initial_q": 0.0,
            "alpha": 0.1
        }
    ]

    # 实验参数
    runs = 2000      # 独立运行次数
    steps = 1000     # 每次运行的步数
    k = 10          # 动作臂数量
    seed = 42       # 随机种子

    # 运行乐观初值对比实验
    print("开始运行乐观初值实验...")
    results = run_optimistic_experiment(
        configs=configs,
        runs=runs,
        steps=steps,
        k=k,
        seed=seed
    )

    # 绘制对比图
    plot_optimistic_curves(
        results=results,
        steps=steps,
        out_prefix="bandit_optimistic"
    )

    # 展示单次运行示例（乐观初值策略）
    print("\n生成乐观初值单次运行示例...")
    plot_single_optimistic_run(
        config=configs[0],  # 使用乐观初值配置
        steps=200,
        out_prefix="optimistic_single_run"
    )

    print("\n=== 实验完成 ===")
    print("图片已保存:")
    print("- bandit_optimistic_avg_reward.png：乐观 vs 现实初值的平均奖励对比")
    print("- bandit_optimistic_optimal_pct.png：乐观 vs 现实初值的最优动作选择率对比")
    print("- optimistic_single_run.png：乐观初值策略的单次运行示例")
    
    print("\n关键洞察:")
    print("🔵 乐观初值策略 (Q₁=5.0, ε=0):")
    print("   - 初期会尝试所有动作（因为所有Q值都很高）")
    print("   - 随着经验积累，Q值逐渐收敛到真实值")
    print("   - 即使是纯贪婪策略也具备了探索能力")
    
    print("⚫ 现实初值策略 (Q₁=0, ε=0.1):")
    print("   - 依赖ε-贪婪进行随机探索")
    print("   - 探索程度恒定（始终有10%概率随机选择）")
    print("   - 可能在局部最优中停滞更久")
    
    print("\n💡 实验结论:")
    print("   乐观初值是一种巧妙的探索技术，通过设置高初值鼓励")
    print("   早期探索，随着学习的进行自然减少探索程度。")


if __name__ == "__main__":
    main()
