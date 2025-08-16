# 强化学习从零开始 (RL Zero to Hero)

强化学习自学实验代码仓库，从多臂强盗问题开始逐步深入强化学习的核心概念。

## 🎯 第四节课：多臂强盗问题

### [📁 lesson04_multi_armed_bandit](./lesson04_multi_armed_bandit/)

**实验目标**：理解强化学习中探索与利用的权衡，对比三种ε-贪婪策略

- **核心算法**：ε-贪婪策略
- **对比策略**：ε = 0（纯贪婪）、ε = 0.01（少量探索）、ε = 0.1（较多探索）
- **关键发现**：
  - 纯贪婪策略容易陷入次优解
  - 适当探索能发现真正的最优动作
  - 探索率影响收敛速度和最终性能

### 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/Dickyzzx/rl-zero-to-hero.git

# 进入实验目录
cd rl-zero-to-hero/lesson04_multi_armed_bandit

# 安装依赖
pip install numpy matplotlib

# 运行实验
python RL_04.py
```

## 📊 实验结果

实验会生成三张对比图表：
- 平均奖励随时间变化
- 最优动作选择率变化
- 单次运行噪声示例

## 🛠 环境要求

- Python 3.7+
- NumPy
- Matplotlib

## 📁 项目结构

```
rl-zero-to-hero/
├── README.md                      # 项目说明
└── lesson04_multi_armed_bandit/   # 第四节课实验
    ├── RL_04.py                   # 主实验代码
    ├── README.md                  # 详细实验说明
    └── *.png                      # 实验结果图表
```

---

**学习笔记**：这是强化学习入门的第一个重要实验，帮助理解探索与利用这一核心概念。
