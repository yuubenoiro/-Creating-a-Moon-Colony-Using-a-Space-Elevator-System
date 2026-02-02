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
    创建多维度决策分析图
    (A) 时间 vs 环境 (B) 成本 vs 升温 (C) 风险趋势分析 (D) 三大方案对比
    """
    # 提取数据
    r_vals = [d['rocket_ratio'] for d in results]
    time_vals = [d['time_years'] for d in results]
    env_vals = [d['co2e_million_tons'] for d in results]
    cost_vals = [d['cost_trillion'] for d in results]
    temp_vals = [d['temp_rise'] for d in results]
    risk_vals = [d['risk_score'] for d in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Problem 4: Multi-Dimensional Strategic Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # ===== 图1: 环境影响 vs 时间 (Pareto Frontier) =====
    ax1 = axes[0, 0]
    sc1 = ax1.scatter(time_vals, env_vals, c=r_vals, cmap='RdYlGn_r', 
                      s=80, edgecolors='black', linewidth=0.5, alpha=0.8)
    ax1.set_xlabel('Project Duration (Years)')
    ax1.set_ylabel('Total CO2e (Million Tons)')
    ax1.set_title('(A) Environmental-Temporal Trade-off', fontweight='bold')
    plt.colorbar(sc1, ax=ax1, label='Rocket Usage Rate')
    ax1.grid(True, linestyle='--', alpha=0.3)

    # ===== 图2: 成本 vs 升温贡献 =====
    ax2 = axes[0, 1]
    sc2 = ax2.scatter(cost_vals, temp_vals, c=r_vals, cmap='RdYlGn_r',
                      s=80, edgecolors='black', linewidth=0.5, alpha=0.8)
    ax2.set_xlabel('Total Cost (Trillion USD)')
    ax2.set_ylabel('Global Temp Rise Contribution (K)')
    ax2.set_title('(B) Cost-Climate Impact Analysis', fontweight='bold')
    plt.colorbar(sc2, ax=ax2, label='Rocket Usage Rate')
    ax2.grid(True, linestyle='--', alpha=0.3)

    # ===== 图3: 综合风险评估 (根据补全的 risk_score 逻辑) =====
    ax3 = axes[1, 0]
    ax3.plot(r_vals, risk_vals, color='darkred', lw=2.5, label='Risk Score')
    ax3.fill_between(r_vals, risk_vals, color='red', alpha=0.1)
    
    ax3.set_xlabel('Rocket Usage Rate (r_usage)')
    ax3.set_ylabel('System Risk Score (0-100)')
    ax3.set_title('(C) Systemic Risk & Reliability Profile', fontweight='bold')
    ax3.set_ylim(0, 105)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.legend()

    # ===== 图4: 归一化对比 (恢复原始三方案对比) =====
    ax4 = axes[1, 1]
    metrics = ['Time', 'Cost', 'CO2e', 'Risk']
    # 选择三个代表性方案进行对比：0%, 50%, 100%
    scenarios = [0, len(results)//2, -1] 
    labels = ['Pure Elevator', 'Balanced (50-50)', 'Pure Rocket']
    colors = ['green', 'blue', 'red']
    
    for i, idx in enumerate(scenarios):
        res = results[idx]
        # 归一化处理：当前值 / 该指标在所有结果中的最大值
        vals = [
            res['time_years'] / max(time_vals), 
            res['cost_trillion'] / max(cost_vals), 
            res['co2e_million_tons'] / max(env_vals), 
            res['risk_score'] / max(risk_vals)
        ]
        ax4.bar(np.arange(len(metrics)) + i*0.25, vals, 0.25, label=labels[i], color=colors[i], alpha=0.7)
    
    ax4.set_xticks(np.arange(len(metrics)) + 0.25)
    ax4.set_xticklabels(metrics)
    ax4.set_ylabel('Normalized Score (Lower is Better)')
    ax4.set_title('(D) Multi-Criteria Comparison', fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.set_ylim(0, 1.2) # 留出图例空间
    ax4.grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✓ Visualization saved: {save_path}")

def environmental_minimization_strategy(results):
    """
    寻找最优策略并打印报告
    """
    # 筛选满足基本时效的方案
    feasible = [r for r in results if r['time_years'] <= 400]
    optimal = min(feasible, key=lambda x: x['co2e_million_tons']) if feasible else results[0]
    
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print(f"Optimal Rocket Usage: {optimal['rocket_ratio']*100:.1f}%")
    print(f"Completion Time:      {optimal['time_years']:.1f} Years")
    print(f"CO2e Emissions:       {optimal['co2e_million_tons']:.2f} Mt")
    print(f"Risk Score:           {optimal['risk_score']:.1f}/100")
    print("="*50)
    return optimal

if __name__ == "__main__":
    p = EnvironmentalParameters()
    model = EnvironmentalImpactModel(p)
    analyzer = DecisionAnalyzer(model)
    
    # 运行 100 个采样点以获得平滑的风险曲线
    results = analyzer.pareto_analysis(100_000_000, n_points=100)
    
    create_comprehensive_visualization(results)
    environmental_minimization_strategy(results)
    
    pd.DataFrame(results).to_csv("decision_results.csv", index=False)