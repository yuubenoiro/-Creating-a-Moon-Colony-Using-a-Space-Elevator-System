import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from constants import EnvironmentalParameters
from engines import EnvironmentalImpactModel
from analyzer3 import DecisionAnalyzerV2

# 绘图风格设置
plt.style.use('seaborn-v0_8-muted')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.grid'] = True

def run_stochastic_study(total_mass=100_000_000, n_points=20):
    """运行分析并生成数据"""
    p = EnvironmentalParameters()
    model = EnvironmentalImpactModel(p)
    analyzer = DecisionAnalyzerV2(model)
    
    print(f"🚀 Starting Monte Carlo Pareto Analysis (n_points={n_points})...")
    # 模拟无自举方案 (alpha=0)
    results = analyzer.pareto_analysis(total_mass, n_points=n_points, alpha=0.0)
    return pd.DataFrame(results)

def create_stochastic_viz(df, save_path='main5_stochastic_analysis.png'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Problem 4: Space Logistics under Stochastic Failure & Reliability Growth', 
                 fontsize=18, fontweight='bold')

    # ===== (1) 时间不确定性分析 (Time with Confidence Band) =====
    ax1 = axes[0, 0]
    # 绘制均值线
    ax1.plot(df['rocket_ratio'], df['time_years'], 'o-', color='navy', label='Mean Duration')
    # 绘制标准差阴影（展示时间波动）
    ax1.fill_between(df['rocket_ratio'], 
                     df['time_years'] - df['time_std'], 
                     df['time_years'] + df['time_std'], 
                     color='navy', alpha=0.2, label='1-Sigma Uncertainty')
    ax1.set_xlabel('Rocket Usage Ratio')
    ax1.set_ylabel('Project Duration (Years)')
    ax1.set_title('A. Temporal Robustness under Random Failure', fontweight='bold')
    ax1.legend()

    # ===== (2) 环境与升温的散点分布 (Risk-Color-Coded) =====
    ax2 = axes[0, 1]
    # 点的大小代表成本，颜色代表风险评分
    sc2 = ax2.scatter(df['time_years'], df['temp_rise'], 
                      c=df['risk_score'], s=df['cost_trillion']*10, 
                      cmap='YlOrRd', edgecolors='black', alpha=0.7)
    ax2.set_xlabel('Mean Completion Time (Years)')
    ax2.set_ylabel('Avg Temperature Rise (K)')
    ax2.set_title('B. Eco-Temporal Trade-off (Size=Cost, Color=Risk)', fontweight='bold')
    plt.colorbar(sc2, ax=ax2, label='System Risk Score')

    # ===== (3) 风险构成分布图 (Stacked Area Plot) =====
    ax3 = axes[1, 0]
    # 我们根据之前的逻辑，简单拆解风险（仅为示意趋势）
    # 在 analyzer2 中，风险 = 波动风险 + 单点风险
    volatility_risk = (df['time_std'] / df['time_years']) * 150
    redundancy_risk = 40 * (1 - 4 * df['rocket_ratio'] * (1 - df['rocket_ratio']))
    
    ax3.stackplot(df['rocket_ratio'], volatility_risk, redundancy_risk, 
                  labels=['Fault Volatility Risk', 'Single-Point Fragility'],
                  colors=['#ff9999','#66b3ff'], alpha=0.8)
    ax3.set_xlabel('Rocket Usage Ratio')
    ax3.set_ylabel('Risk Contribution Weight')
    ax3.set_title('C. Risk Source Decomposition', fontweight='bold')
    ax3.legend(loc='upper right')

    # ===== (4) 关键方案的雷达/雷达替代图 (Violin-like Boxplot) =====
    ax4 = axes[1, 1]
    # 对比 0%, 50%, 100% 方案的归一化性能
    compare_indices = [0, len(df)//2, len(df)-1]
    compare_df = df.iloc[compare_indices].copy()
    
    # 转换数据以便绘制
    metrics = ['time_years', 'cost_trillion', 'co2e_million_tons', 'risk_score']
    melted = compare_df.melt(id_vars='rocket_ratio', value_vars=metrics)
    # 归一化各指标以便在同一张图对比
    for m in metrics:
        melted.loc[melted['variable'] == m, 'value'] /= df[m].max()

    sns.barplot(data=melted, x='variable', y='value', hue='rocket_ratio', ax=ax4, palette='viridis')
    ax4.set_title('D. Normalized Strategic Comparison', fontweight='bold')
    ax4.set_ylabel('Normalized Score (Lower is Better)')
    ax4.set_xticklabels(['Time', 'Cost', 'CO2e', 'Risk'])
    ax4.legend(title='Rocket Ratio', loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    print(f"✅ Stochastic Visualization saved to {save_path}")

if __name__ == "__main__":
    # 1. 运行分析
    results_df = run_stochastic_study(n_points=25)
    
    # 2. 生成图表
    create_stochastic_viz(results_df)
    
    # 3. 统计关键发现
    best_risk_idx = results_df['risk_score'].idxmin()
    print("\n" + "="*50)
    print("STOCHASTIC OPTIMIZATION FINDINGS")
    print("-" * 50)
    print(f"Robust Optimum (Min Risk): r = {results_df.loc[best_risk_idx, 'rocket_ratio']:.2f}")
    print(f"Time Expectation: {results_df.loc[best_risk_idx, 'time_years']:.1f} ± {results_df.loc[best_risk_idx, 'time_std']:.1f} years")
    print(f"Environmental Impact: {results_df.loc[best_risk_idx, 'co2e_million_tons']:.2f} Mt CO2e")
    print("="*50)