import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model3 import stochastic_sim_v4

# ==========================================
# 1. 蒙特卡洛数据采集 (Alpha=0)
# ==========================================
def get_cdf_data(iterations=2000):
    print(f">>> Running {iterations} Monte Carlo iterations for Scenario: Alpha=0...")
    # 模拟 Alpha=0, Rocket Usage=0.4 的情况
    results = [stochastic_sim_v4([0.0, 0.4]) for _ in range(iterations)]
    return np.sort([r[0] for r in results])

# ==========================================
# 2. 绘图与美化 (O奖论文标准)
# ==========================================
def plot_o_award_cdf(data):
    # 设置论文专用样式
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_context("paper", font_scale=1.2)
    plt.figure(figsize=(10, 6), dpi=300)

    # 计算累积概率
    y = np.arange(1, len(data) + 1) / len(data)
    
    # 核心曲线
    plt.plot(data, y, color='#1f4e79', linewidth=2.5, label='Cumulative Probability (CDF)')
    
    # 填充概率区间 (25% - 75% 核心概率区)
    q25, q50, q75 = np.percentile(data, [25, 50, 75])
    plt.fill_between(data, 0, y, where=(data >= q25) & (data <= q75), 
                     color='#1f4e79', alpha=0.1, label='Interquartile Range (IQR)')

    # 绘制 95% 置信保证线 (MCM关键点)
    p95 = np.percentile(data, 95)
    plt.axhline(0.95, color='#c00000', linestyle='--', linewidth=1)
    plt.axvline(p95, color='#c00000', linestyle='--', linewidth=1)
    plt.scatter(p95, 0.95, color='#c00000', zorder=5)
    plt.text(p95 + 1, 0.92, f'95% Guarantee: {p95:.1f}y', color='#c00000', fontweight='bold')

    # 绘制中位数线
    plt.axhline(0.5, color='gray', linestyle=':', linewidth=1)
    plt.axvline(q50, color='gray', linestyle=':', linewidth=1)
    plt.text(data[0], 0.52, f'Median: {q50:.1f}y', color='gray')

    # 图表细节 (符合论文规范)
    plt.title("Figure X: Cumulative Probability of Mission Success (Alpha=0)", 
              fontsize=15, fontweight='bold', pad=20)
    plt.xlabel("Years from Deployment (Beginning in 2050)", fontsize=12)
    plt.ylabel("Probability of Reaching 100M Tons ", fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='lower right', frameon=True, edgecolor='black')
    
    # 设置坐标轴范围
    plt.xlim(data.min() - 2, data.max() + 2)
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("O_Award_Style_CDF.png")
    plt.show()

if __name__ == "__main__":
    t_data = get_cdf_data()
    plot_o_award_cdf(t_data)