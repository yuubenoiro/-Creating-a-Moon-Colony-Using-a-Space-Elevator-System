import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from constants import EnvironmentalParameters
from engines import EnvironmentalImpactModel
from analyzers import DecisionAnalyzer

# 设置学术绘图风格
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

def create_comprehensive_visualization(results, save_path='problem4_analysis.png'):
    """
    创建多维度决策分析图（2x2子图布局）
    """
    # 提取数据
    r_vals = [d['rocket_ratio'] for d in results]
    time_vals = [d['time'] for d in results] # 对应 analyzer 中的 key
    env_vals = [d['co2e'] for d in results]
    cost_vals = [d['cost'] for d in results]
    temp_vals = [d['temp_rise'] for d in results]
    
    # 模拟风险评分（analyzer中若未实现可在此简化计算，此处假定 analyzer 已包含风险评分）
    # 如果你的 analyzers.py 还没写风险评分逻辑，这里做一个映射
    risk_vals = [50 * (1 - 4 * r * (1 - r)) + (10 if r < 0.2 else 0) for r in r_vals]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Problem 4: Multi-Dimensional Environmental & Strategic Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # ===== 图1: 环境影响 vs 时间 =====
    ax1 = axes[0, 0]
    sc1 = ax1.scatter(time_vals, env_vals, c=r_vals, cmap='RdYlGn_r', 
                      s=100, edgecolors='black', linewidth=0.5, alpha=0.8)
    ax1.set_xlabel('Project Duration (Years)')
    ax1.set_ylabel('Total CO2e (Million Tons)')
    ax1.set_title('(A) Environmental-Temporal Trade-off', fontweight='bold')
    plt.colorbar(sc1, ax=ax1, label='Rocket Ratio')
    ax1.grid(True, linestyle='--', alpha=0.3)

    # ===== 图2: 成本 vs 升温贡献 =====
    ax2 = axes[0, 1]
    sc2 = ax2.scatter(cost_vals, temp_vals, c=r_vals, cmap='RdYlGn_r',
                      s=100, edgecolors='black', linewidth=0.5, alpha=0.8)
    ax2.set_xlabel('Total Cost (Trillion USD)')
    ax2.set_ylabel('Global Temp Rise (K)')
    ax2.set_title('(B) Cost-Climate Trade-off', fontweight='bold')
    plt.colorbar(sc2, ax=ax2, label='Rocket Ratio')
    ax2.grid(True, linestyle='--', alpha=0.3)

    # ===== 图3: 发射规模分析 =====
    ax3 = axes[1, 0]
    # 假设每发射消耗4500吨燃料，BC排放约150kg (基于 constants.py)
    launches = [(d['rocket_ratio'] * 100_000_000 / 125) for d in results]
    ax3.plot(r_vals, launches, color='navy', lw=2, label='Total Launches')
    ax3.set_xlabel('Rocket Usage Ratio')
    ax3.set_ylabel('Number of Launches')
    ax3.set_title('(C) Logistics Scale Analysis', fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.3)

    # ===== 图4: 归一化对比 (Bar Chart) =====
    ax4 = axes[1, 1]
    metrics = ['Time', 'Cost', 'CO2e', 'TempRise']
    scenarios = [0, len(results)//2, -1] # 选 0%, 50%, 100% 比例
    labels = ['Pure Elevator', 'Balanced', 'Pure Rocket']
    colors = ['green', 'blue', 'red']
    
    for i, idx in enumerate(scenarios):
        res = results[idx]
        # 极简归一化处理用于展示
        vals = [res['time']/max(time_vals), res['cost']/max(cost_vals), 
                res['co2e']/max(env_vals), res['temp_rise']/max(temp_vals)]
        ax4.bar(np.arange(len(metrics)) + i*0.25, vals, 0.25, label=labels[i], color=colors[i], alpha=0.7)
    
    ax4.set_xticks(np.arange(len(metrics)) + 0.25)
    ax4.set_xticklabels(metrics)
    ax4.set_title('(D) Normalized Multi-Criteria Comparison', fontweight='bold')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✓ Visualization saved: {save_path}")

def environmental_minimization_strategy(results):
    """
    在约束条件下寻找最优绿色策略
    """
    # 约束条件：时间 < 400年，成本 < 15 Trillion
    feasible = [r for r in results if r['time'] <= 400 and r['cost'] <= 15]
    
    if not feasible:
        optimal = min(results, key=lambda x: x['co2e']) # 降级方案
    else:
        optimal = min(feasible, key=lambda x: x['co2e'])
    
    print("\n" + "="*50)
    print("STRATEGY RECOMMENDATION (Green Focus)")
    print(f"Rocket Ratio: {optimal['rocket_ratio']*100:.1f}%")
    print(f"Time: {optimal['time']:.1f} Years")
    print(f"CO2e: {optimal['co2e']:.2f} Million Tons")
    print(f"Temp Rise: {optimal['temp_rise']:.4f} K")
    print("="*50)
    return optimal

if __name__ == "__main__":
    # 1. 初始化
    p = EnvironmentalParameters()
    model = EnvironmentalImpactModel(p)
    analyzer = DecisionAnalyzer(model)
    
    # 2. 运行帕累托分析 (1亿吨)
    results = analyzer.pareto_analysis(100_000_000, n_points=50)
    
    # 3. 可视化
    create_comprehensive_visualization(results)
    
    # 4. 策略输出
    environmental_minimization_strategy(results)
    
    # 5. 导出
    pd.DataFrame(results).to_csv("decision_results.csv", index=False)