import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# ==========================================
# 1. 提取 model2.py 中的参数
# ==========================================
YEAR_START = 2050
YEAR_END = 2100
YEARS = np.arange(YEAR_START, YEAR_END + 1)
T = YEARS - YEAR_START

# 你的模型参数
LAMBDA_SMALL_INIT = 20.0
DECAY_RATE = 0.05
LAMBDA_BIG_FIXED = 0.3

# ==========================================
# 2. 计算理论期望与随机模拟
# ==========================================
# 期望故障率 lambda(t) = lambda_small(t) + lambda_big
expected_lambda = LAMBDA_SMALL_INIT * np.exp(-DECAY_RATE * T) + LAMBDA_BIG_FIXED

# 基于泊松分布生成单次随机模拟场景
np.random.seed(42)  # 固定种子以获得一致的可视化效果
simulated_scenarios = np.random.poisson(expected_lambda)

# 计算 95% 泊松置信区间 (Confidence Interval)
# 使用 ppf (百分位数函数) 得到区间上下界
ci_lower = poisson.ppf(0.025, expected_lambda)
ci_upper = poisson.ppf(0.975, expected_lambda)

# ==========================================
# 3. 绘制专业学术图表
# ==========================================
plt.figure(figsize=(12, 6), dpi=120)

# 绘制 95% 置信区间阴影 (体现方差随 lambda 减小而收缩)
plt.fill_between(YEARS, ci_lower, ci_upper, color='#ff6b6b', alpha=0.15, 
                 label='95% Poisson Confidence Interval')

# 绘制期望故障率曲线 (红色虚线)
plt.plot(YEARS, expected_lambda, color='#a80000', linestyle='--', linewidth=2, 
         label='Expected Failure Rate ($\lambda$)')

# 绘制单次随机模拟结果 (带点的实线)
plt.plot(YEARS, simulated_scenarios, color='black', marker='o', markersize=4, 
         linewidth=1, alpha=0.8, label='Simulated Scenario')

# 添加关键阶段标注
plt.text(2052, 28, 'High Risk & High Variance\n(Early Phase)', 
         color='#a80000', fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
plt.text(2090, 4.5, 'Operational Stability\n(Mature Phase)', 
         color='#2d6a4f', fontsize=11, fontweight='bold')

# 绘制趋势趋势箭头 (从高波动指向稳定)
plt.annotate('', xy=(2080, 5), xytext=(2055, 18),
             arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=6, alpha=0.6))

# 图表装饰
plt.title("Reliability Growth: The Dissipation of Risk (2050-2100)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Annual Number of Breakdowns", fontsize=12)
plt.grid(True, which='both', linestyle=':', alpha=0.4)
plt.legend(loc='upper right', frameon=True)
plt.xlim(YEAR_START, YEAR_END)
plt.ylim(0, 32)

plt.tight_layout()
plt.savefig("Reliability_Growth_Model2.png")
plt.show()