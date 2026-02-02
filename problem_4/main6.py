import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from constants import EnvironmentalParameters
from engines import EnvironmentalImpactModel
from analyzer4 import DecisionAnalyzerV4 as DecisionAnalyzerV2

# 绘图风格设置
plt.style.use('seaborn-v0_8-muted')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.grid'] = True

def run_stochastic_study(total_mass=100_000_000, n_points=25):
    """运行集成动态Eta模型的蒙特卡洛分析"""
    p = EnvironmentalParameters()
    model = EnvironmentalImpactModel(p)
    analyzer = DecisionAnalyzerV2(model)
    
    print(f"🚀 Running Integrated Stochastic Analysis (n_points={n_points})...")
    # 模拟分析，analyzer2 内部现在会自动调用 engines 里的动态 eta 计算
    results = analyzer.pareto_analysis(total_mass, n_points=n_points, alpha=0.0)
    return pd.DataFrame(results)

def create_stochastic_viz(df, save_path='main6_feedback_analysis.png'):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Problem 4: Space Logistics with BC-Induced Efficiency Feedback', 
                 fontsize=18, fontweight='bold')

    # ===== (1) 时间响应曲线 (反映效率崩塌导致的时间拐点) =====
    ax1 = axes[0, 0]
    ax1.plot(df['rocket_ratio'], df['time_years'], 'o-', color='navy', label='Mean Duration')
    ax1.fill_between(df['rocket_ratio'], 
                     df['time_years'] - df['time_std'], 
                     df['time_years'] + df['time_std'], 
                     color='navy', alpha=0.2)
    ax1.set_xlabel('Rocket Usage Ratio ($r$)')
    ax1.set_ylabel('Total Project Duration (Years)')
    ax1.set_title('A. Temporal Impact of Efficiency Feedback', fontweight='bold')
    
    # 标注可能的瓶颈点
    optimal_time_idx = df['time_years'].idxmin()
    ax1.annotate('Potential Saturation', xy=(df.loc[optimal_time_idx, 'rocket_ratio'], df.loc[optimal_time_idx, 'time_years']),
                 xytext=(0.5, 0.8), textcoords='axes fraction',
                 arrowprops=dict(facecolor='black', shrink=0.05))

    # ===== (2) 成本-升温-风险 三维关系 =====
    ax2 = axes[0, 1]
    sc2 = ax2.scatter(df['cost_trillion'], df['temp_rise'], 
                      c=df['risk_score'], s=df['time_years']*2, 
                      cmap='plasma', edgecolors='black', alpha=0.7)
    ax2.set_xlabel('Total Cost (Trillion USD)')
    ax2.set_ylabel('Global Temp Rise (K)')
    ax2.set_title('B. Cost-Climate-Risk Surface (Size=Duration)', fontweight='bold')
    plt.colorbar(sc2, ax=ax2, label='System Risk Score')

    # ===== (3) 核心反馈图：黑碳强度 vs 电梯效率 η (新图) =====
    ax3 = axes[1, 0]
    # 我们从引擎重新生成一组干净的 eta 曲线用于展示
    p = EnvironmentalParameters()
    model = EnvironmentalImpactModel(p)
    x_test = np.linspace(0, 1, 100)
    eta_test = [model.calculate_dynamic_eta(x * 5.0) for x in x_test]
    
    
    ax3.plot(x_test, eta_test, 'r-', lw=3, label='Dynamic Efficiency $\eta$')
    ax3.fill_between(x_test, 0.1, eta_test, color='red', alpha=0.1)
    
    # 叠加 AOD 增长趋势（归一化）
    ax3_twin = ax3.twinx()
    aod_trend = [model.calculate_bc_pollution_index(x * 100) for x in x_test]
    ax3_twin.plot(x_test, aod_trend, 'g--', alpha=0.6, label='AOD (Pollution)')
    
    ax3.set_xlabel('Rocket Usage Ratio ($r$)')
    ax3.set_ylabel('Elevator Efficiency $\eta$', color='r')
    ax3_twin.set_ylabel('AOD / Stratosphere Disturbance', color='g')
    ax3.set_title('C. Efficiency Collapse via Atmospheric Feedback', fontweight='bold')
    ax3.set_ylim(0, 1.0)

    # ===== (4) 策略敏感度分析 (归一化对比) =====
    ax4 = axes[1, 1]
    # 选三个点对比：环保型(r=0), 均衡型(r=0.2), 激进型(r=1.0)
    compare_r = [0.0, 0.2, 1.0]
    # 找到离这些值最近的索引
    compare_indices = [ (df['rocket_ratio']-r).abs().idxmin() for r in compare_r ]
    compare_df = df.iloc[compare_indices].copy()
    
    metrics = ['time_years', 'cost_trillion', 'co2e_million_tons', 'risk_score']
    melted = compare_df.melt(id_vars='rocket_ratio', value_vars=metrics)
    # 归一化
    for m in metrics:
        melted.loc[melted['variable'] == m, 'value'] /= df[m].max()

    sns.barplot(data=melted, x='variable', y='value', hue='rocket_ratio', ax=ax4, palette='magma')
    ax4.set_title('D. Comparative Performance Profile', fontweight='bold')
    ax4.set_xticklabels(['Time', 'Cost', 'CO2e', 'Risk'])
    ax4.set_ylim(0, 1.2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    print(f"✅ Integrated Feedback Visualization saved to {save_path}")

if __name__ == "__main__":
    # 运行
    results_df = run_stochastic_study(n_points=30)
    create_stochastic_viz(results_df)
    
    # 输出基于反馈模型的建议
    # 此时的“最优”可能不再是 r=1，因为 r=1 时效率太低
    best_time_idx = results_df['time_years'].idxmin()
    print("\n" + "="*50)
    print("FEEDBACK-AWARE OPTIMIZATION")
    print("-" * 50)
    print(f"Optimal Rocket Ratio: r = {results_df.loc[best_time_idx, 'rocket_ratio']:.2f}")
    print(f"Expected Duration:    {results_df.loc[best_time_idx, 'time_years']:.1f} years")
    print(f"Efficiency at Opt:    η ≈ {EnvironmentalImpactModel(EnvironmentalParameters()).calculate_dynamic_eta(results_df.loc[best_time_idx, 'rocket_ratio']*5):.3f}")
    print("="*50)