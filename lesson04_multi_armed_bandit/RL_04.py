# filename: bandit_eps_greedy.py
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


class EpsilonGreedyAgent:
    """
    ε-Greedy 策略智能体类

    输入参数:
    - k (int): 动作臂数量
    - epsilon (float): 探索率 ε（以 ε 的概率随机探索）
    - rng (np.random.Generator, 默认=None): 随机数生成器

    输出:
    - select_action() -> int: 按 ε-Greedy 规则选择动作
    - update(action, reward) -> None: 用样本平均公式更新 Q 值
    - reset() -> None: 重置 Q 值和计数器

    用途:
    - 在 k-臂老虎机环境中平衡探索与利用，并在线估计各动作的价值 Q(a)
    """

    def __init__(self, k: int, epsilon: float, rng: np.random.Generator = None):
        self.k = k                                   # 动作臂数量
        self.epsilon = epsilon                       # 探索率 ε
        self.rng = rng if rng is not None else np.random.default_rng()  # 随机数生成器
        self.reset()                                 # 初始化 Q 和 N

    def reset(self):
        self.Q = np.zeros(self.k, dtype=float)       # 动作价值估计 Q(a)，初始为 0
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
        # 作用: 用增量平均公式更新该动作的价值估计 Q(a)
        self.N[action] += 1                          # 该动作计数 +1
        step_size = 1.0 / self.N[action]             # 步长 α_t = 1 / N(a)
        self.Q[action] += step_size * (reward - self.Q[action])  # 增量更新 Q(a)


def run_single(agent: EpsilonGreedyAgent, bandit: TenArmedBandit, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    单次运行（Single Run）

    输入参数:
    - agent (EpsilonGreedyAgent): ε-Greedy 智能体
    - bandit (TenArmedBandit): 10-臂老虎机环境
    - steps (int): 运行的时间步数

    输出:
    - rewards (np.ndarray[steps]): 每一步的即时奖励
    - optimal_taken (np.ndarray[steps]): 每一步是否选到最优动作（0/1）

    用途:
    - 执行一次完整的策略交互过程，用于展示单次曲线的噪声特性或调试
    """
    rewards = np.zeros(steps, dtype=float)           # 预分配奖励数组
    optimal_taken = np.zeros(steps, dtype=int)       # 预分配是否最优动作的标记数组

    agent.reset()                                    # 重置智能体（清空 Q 与 N）
    bandit.reset()                                   # 重置环境（重新抽取 q*(a)）

    for t in range(steps):                           # 遍历每个时间步
        a = agent.select_action()                    # 选择一个动作
        r = bandit.step(a)                           # 执行动作并获得奖励
        agent.update(a, r)                           # 用奖励更新 Q(a)
        rewards[t] = r                               # 记录该步的奖励
        optimal_taken[t] = 1 if a == bandit.optimal_action else 0  # 记录是否为最优动作

    return rewards, optimal_taken                    # 返回两个序列


def run_experiment(
    epsilons: List[float],
    runs: int,
    steps: int,
    k: int = 10,
    q_mean: float = 0.0,
    q_std: float = 1.0,
    reward_std: float = 1.0,
    seed: int = 42
):
    """
    多次独立运行并取平均（Averaging over Independent Runs）

    输入参数:
    - epsilons (List[float]): 要比较的 ε 值列表（如 [0, 0.01, 0.1]）
    - runs (int): 独立运行次数（例如 2000 次，用于统计平均）
    - steps (int): 每次运行的时间步数（例如 1000）
    - k (int, 默认=10): 动作臂数量
    - q_mean (float, 默认=0.0): q*(a) 总体均值
    - q_std (float, 默认=1.0): q*(a) 总体标准差
    - reward_std (float, 默认=1.0): 即时奖励标准差
    - seed (int, 默认=42): 主随机种子（用于生成各 run 的派生种子）

    输出:
    - results (dict[float, Tuple[np.ndarray, np.ndarray]]):
        {epsilon: (avg_rewards[steps], avg_optimal_pct[steps])}
        其中 avg_optimal_pct 是最优动作选择率（百分比）

    用途:
    - 对每个 ε 进行多次独立试验，返回平均奖励曲线和平均最优动作选择率曲线，用于科学比较
    """
    rng_master = np.random.default_rng(seed)         # 主随机源，用来派生每次 run 的子种子
    results = {}                                     # 用字典保存不同 ε 的平均结果

    for eps in epsilons:                             # 遍历每个 ε
        avg_rewards = np.zeros(steps, dtype=float)   # 累加奖励（待会儿除以 runs 取平均）
        avg_optimal = np.zeros(steps, dtype=float)   # 累加是否为最优（待会儿除以 runs 再转百分比）

        for i in range(runs):                        # 独立运行 runs 次
            run_seed_env = int(rng_master.integers(0, 2**31 - 1))       # 为“环境”生成派生种子
            run_seed_agent = int(rng_master.integers(0, 2**31 - 1))     # 为“智能体”生成派生种子
            rng_env = np.random.default_rng(run_seed_env)               # 环境的随机源
            rng_agent = np.random.default_rng(run_seed_agent)           # 智能体的随机源

            bandit = TenArmedBandit(k=k, q_mean=q_mean, q_std=q_std, reward_std=reward_std, rng=rng_env)  # 构造环境
            agent = EpsilonGreedyAgent(k=k, epsilon=eps, rng=rng_agent)                                     # 构造智能体

            rewards, optimal_taken = run_single(agent, bandit, steps)  # 跑一次得到两个序列
            avg_rewards += rewards                   # 累加奖励
            avg_optimal += optimal_taken             # 累加是否为最优

        avg_rewards /= runs                          # 将奖励累积值除以 runs 得到平均奖励
        avg_optimal = (avg_optimal / runs) * 100.0   # 将是否为最优的频率转为百分比
        results[eps] = (avg_rewards, avg_optimal)   # 存入结果字典

    return results                                   # 返回所有 ε 的平均结果


def plot_curves(results, steps: int, out_prefix: str = "bandit_eps"):
    """
    绘制平均曲线（两张图）

    输入参数:
    - results (dict): run_experiment 的返回结果 {ε: (avg_rewards, avg_optimal_pct)}
    - steps (int): 时间步数（横轴长度）
    - out_prefix (str, 默认="bandit_eps"): 输出图片文件名前缀

    输出:
    - 保存两张图片：
      1) {out_prefix}_avg_reward.png —— 平均奖励曲线
      2) {out_prefix}_optimal_pct.png —— 平均最优动作选择率曲线

    用途:
    - 把多次独立运行的平均结果可视化，便于科学比较算法
    """
    t = np.arange(steps)                             # 生成横轴坐标 0..steps-1

    # 图1：平均奖励
    plt.figure(figsize=(8, 5))                       # 新建图像，设置大小
    for eps, (avg_rewards, _) in results.items():    # 遍历不同 ε 的结果
        plt.plot(t, avg_rewards, label=f"epsilon={eps}")  # 画出奖励曲线
    plt.title("Average Reward over Time (10-armed bandit)")  # 标题
    plt.xlabel("Time step")                          # X 轴标签
    plt.ylabel("Average Reward")                     # Y 轴标签
    plt.legend()                                     # 图例
    plt.grid(True, linestyle="--", linewidth=0.5)    # 网格线
    fname1 = f"{out_prefix}_avg_reward.png"          # 输出文件名
    plt.tight_layout()                               # 紧凑布局
    plt.savefig(fname1, dpi=150)                     # 保存图片
    print(f"Saved: {fname1}")                        # 打印保存路径

    # 图2：最优动作选择率
    plt.figure(figsize=(8, 5))                       # 新建图像
    for eps, (_, avg_optimal_pct) in results.items():  # 遍历不同 ε
        plt.plot(t, avg_optimal_pct, label=f"epsilon={eps}")  # 画出最优动作选择率曲线
    plt.title("Optimal Action % over Time (10-armed bandit)")  # 标题
    plt.xlabel("Time step")                          # X 轴标签
    plt.ylabel("Optimal Action (%)")                 # Y 轴标签
    plt.legend()                                     # 图例
    plt.grid(True, linestyle="--", linewidth=0.5)    # 网格线
    fname2 = f"{out_prefix}_optimal_pct.png"         # 输出文件名
    plt.tight_layout()                               # 紧凑布局
    plt.savefig(fname2, dpi=150)                     # 保存图片
    print(f"Saved: {fname2}")                        # 打印保存路径


def plot_single_run(agent: EpsilonGreedyAgent, bandit: TenArmedBandit, steps: int, out_prefix: str = "bandit_eps_single"):
    """
    绘制一次独立运行的奖励曲线（Single Run）

    输入:
    - agent (EpsilonGreedyAgent): 智能体
    - bandit (TenArmedBandit): 老虎机环境
    - steps (int): 时间步数
    - out_prefix (str, 默认="bandit_eps_single"): 输出文件名前缀

    输出:
    - 保存一张图片：{out_prefix}.png —— 单次运行奖励曲线

    用途:
    - 直观展示单次运行曲线的"抖动与噪声"，说明单次曲线不适合比较算法
    """
    rewards, _ = run_single(agent, bandit, steps)  # 单次运行，获得每步奖励序列
    t = np.arange(steps)                           # 横轴时间步数组

    plt.figure(figsize=(8, 5))                     # 新建图像
    plt.plot(t, rewards, linewidth=1)              # 绘制单次奖励曲线（不指定颜色，遵循默认主题）
    plt.title("Reward in a Single Run (noisy example)")  # 标题
    plt.xlabel("Time step")                        # X 轴标签
    plt.ylabel("Reward")                           # Y 轴标签
    plt.grid(True, linestyle="--", linewidth=0.5)  # 网格线
    fname = f"{out_prefix}.png"                    # 输出文件名
    plt.tight_layout()                             # 紧凑布局
    plt.savefig(fname, dpi=150)                    # 保存图片
    print(f"Saved: {fname}")                       # 打印保存路径


def run_comparison_experiment(
    epsilon: float = 0.1,
    runs_list: List[int] = [1, 20, 200, 2000],
    steps: int = 1000,
    k: int = 10,
    q_mean: float = 0.0,
    q_std: float = 1.0,
    reward_std: float = 1.0,
    seed: int = 42
):
    """
    对比不同运行次数的实验效果

    输入参数:
    - epsilon (float): 使用的ε值
    - runs_list (List[int]): 要对比的运行次数列表
    - steps (int): 每次运行的时间步数
    - k (int): 动作臂数量
    - q_mean (float): q*(a) 总体均值
    - q_std (float): q*(a) 总体标准差
    - reward_std (float): 即时奖励标准差
    - seed (int): 主随机种子

    输出:
    - results (dict): {runs: (avg_rewards, avg_optimal_pct)}

    用途:
    - 展示随着运行次数增加，平均曲线如何变得更加稳定和可靠
    """
    results = {}
    rng_master = np.random.default_rng(seed)

    for runs in runs_list:
        print(f"运行 {runs} 次实验...")
        
        avg_rewards = np.zeros(steps, dtype=float)
        avg_optimal = np.zeros(steps, dtype=float)
        
        for i in range(runs):
            run_seed_env = int(rng_master.integers(0, 2**31 - 1))
            run_seed_agent = int(rng_master.integers(0, 2**31 - 1))
            rng_env = np.random.default_rng(run_seed_env)
            rng_agent = np.random.default_rng(run_seed_agent)
            
            bandit = TenArmedBandit(k=k, q_mean=q_mean, q_std=q_std, reward_std=reward_std, rng=rng_env)
            agent = EpsilonGreedyAgent(k=k, epsilon=epsilon, rng=rng_agent)
            
            rewards, optimal_taken = run_single(agent, bandit, steps)
            avg_rewards += rewards
            avg_optimal += optimal_taken
        
        avg_rewards /= runs
        avg_optimal = (avg_optimal / runs) * 100.0
        results[runs] = (avg_rewards, avg_optimal)
    
    return results


def plot_runs_comparison(results, steps: int, epsilon: float, out_prefix: str = "runs_comparison"):
    """
    绘制不同运行次数的对比图

    输入参数:
    - results (dict): run_comparison_experiment 的返回结果
    - steps (int): 时间步数
    - epsilon (float): 使用的ε值
    - out_prefix (str): 输出图片文件名前缀

    输出:
    - 保存两张对比图：平均奖励和最优动作选择率
    """
    t = np.arange(steps)
    
    # 图1：不同运行次数的平均奖励对比
    plt.figure(figsize=(10, 6))
    for runs, (avg_rewards, _) in results.items():
        plt.plot(t, avg_rewards, label=f"{runs} runs", linewidth=2)
    plt.title(f"Average Reward Comparison with Different Number of Runs (ε={epsilon})")
    plt.xlabel("Time Step")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    fname1 = f"{out_prefix}_avg_reward.png"
    plt.tight_layout()
    plt.savefig(fname1, dpi=150)
    print(f"已保存: {fname1}")
    
    # 图2：不同运行次数的最优动作选择率对比
    plt.figure(figsize=(10, 6))
    for runs, (_, avg_optimal_pct) in results.items():
        plt.plot(t, avg_optimal_pct, label=f"{runs} runs", linewidth=2)
    plt.title(f"Optimal Action % Comparison with Different Number of Runs (ε={epsilon})")
    plt.xlabel("Time Step")
    plt.ylabel("Optimal Action (%)")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    fname2 = f"{out_prefix}_optimal_pct.png"
    plt.tight_layout()
    plt.savefig(fname2, dpi=150)
    print(f"已保存: {fname2}")
    
    plt.show()  # 显示图像


def main():
    """
    主程序入口：对比三种不同的ε-贪婪策略

    体现三种策略:
    - ε = 0（纯贪婪）：完全利用，不进行探索
    - ε = 0.01（少量探索）：99%利用，1%探索
    - ε = 0.1（较多探索）：90%利用，10%探索

    输出:
    - bandit_eps_avg_reward.png —— 三种ε策略的平均奖励对比
    - bandit_eps_optimal_pct.png —— 三种ε策略的最优动作选择率对比
    """
    print("=== 10臂强盗问题：三种ε-贪婪策略对比 ===")
    print("正在对比以下三种策略：")
    print("- ε = 0（纯贪婪）：完全利用，不进行探索")
    print("- ε = 0.01（少量探索）：99%利用，1%探索") 
    print("- ε = 0.1（较多探索）：90%利用，10%探索")
    print()
    
    # 实验参数设置
    epsilons = [0, 0.01, 0.1]         # 三种不同的ε值
    runs = 2000                       # 独立运行次数
    steps = 1000                      # 每次运行的步数
    k = 10                           # 动作臂数量
    seed = 42                        # 随机种子
    
    # 运行ε-贪婪策略对比实验
    print("开始运行实验...")
    epsilon_results = run_experiment(
        epsilons=epsilons,
        runs=runs,
        steps=steps,
        k=k,
        seed=seed
    )
    
    # 绘制三种ε策略的对比图
    plot_curves(
        results=epsilon_results,
        steps=steps,
        out_prefix="bandit_eps"
    )
    
    # 额外展示单次运行的噪声特性（使用ε=0.1）
    print("\n正在生成单次运行示例...")
    rng_demo = np.random.default_rng(123)
    demo_bandit = TenArmedBandit(k=k, rng=rng_demo)
    demo_agent = EpsilonGreedyAgent(k=k, epsilon=0.1, rng=rng_demo)
    plot_single_run(demo_agent, demo_bandit, steps=200, out_prefix="bandit_eps_single_run")
    
    print("\n=== 实验完成 ===")
    print("图片已保存:")
    print("- bandit_eps_avg_reward.png：三种ε策略的平均奖励对比")
    print("- bandit_eps_optimal_pct.png：三种ε策略的最优动作选择率对比")
    print("- bandit_eps_single_run.png：单次运行示例（展示噪声特性）")
    print("\n策略分析：")
    print("- ε=0（纯贪婪）：短期内可能表现较好，但容易陷入次优动作")
    print("- ε=0.01（少量探索）：在保持较高利用的同时进行少量探索")
    print("- ε=0.1（较多探索）：通过更多探索找到更好的动作，长期表现更佳")


if __name__ == "__main__":
    main()  # 主程序入口
