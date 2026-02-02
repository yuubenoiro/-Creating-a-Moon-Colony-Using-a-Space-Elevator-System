import numpy as np
import matplotlib.pyplot as plt
from model2 import stochastic_sim_v4 # 确保函数名与你代码一致

def plot_stochastic_pareto(iterations=50):
    print(">>> 正在生成随机帕累托前沿数据，请稍候...")
    
    # 设定参数搜索范围 (固定 Alpha=0, 调整 Rocket Usage)
    r_usage_range = np.linspace(0.1, 1.0, 15)
    all_times = []
    all_costs = []
    
    # 对每一个决策点进行多次模拟取均值
    for r in r_usage_range:
        t_samples = []
        c_samples = []
        for _ in range(iterations):
            t, c = stochastic_sim_v4([0.0, r])
            t_samples.append(t)
            c_samples.append(c / 1e12) # 转换为万亿
        all_times.append(np.mean(t_samples))
        all_costs.append(np.mean(c_samples))

    # 绘制图像
    plt.figure(figsize=(10, 6), dpi=120)
    
    # 1. 绘制前沿曲线
    plt.plot(all_times, all_costs, 'o-', color='#2c3e50', linewidth=2, label='Pareto Frontier (Stochastic Mean)')
    
    # 2. 标注极端点
    plt.annotate(f'Fastest Case\n(Rocket 100%)', xy=(all_times[-1], all_costs[-1]), 
                 xytext=(all_times[-1]+5, all_costs[-1]+2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1))
    
    plt.annotate(f'Economic Case\n(Rocket 10%)', xy=(all_times[0], all_costs[0]), 
                 xytext=(all_times[0]-30, all_costs[0]+5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    # 3. 设置背景区域 (不可行域与次优域)
    plt.fill_between(all_times, all_costs, plt.gca().get_ylim()[1], color='gray', alpha=0.1, label='Suboptimal Region')

    # 图表美化
    plt.title("Pareto Frontier under Stochastic Failures (Alpha=0)", fontsize=14, fontweight='bold')
    plt.xlabel("Average Completion Time (Years)", fontsize=12)
    plt.ylabel("Average Total Cost (Trillion USD)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("Stochastic_Pareto_Frontier.png")
    plt.show()

if __name__ == "__main__":
    plot_stochastic_pareto()