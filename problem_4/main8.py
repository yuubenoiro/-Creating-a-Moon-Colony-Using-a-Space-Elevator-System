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


def save_individual_plots(df, model):
    """分别保存四个分析图表"""
    
    # 统一设置
    plt.style.use('seaborn-v0_8-muted')
    plt.rcParams['font.family'] = 'serif'

    # ==========================================
    # Plot A: Temporal Impact (带最优值标注)
    # ==========================================
    plt.figure(figsize=(10, 7))
    plt.plot(df['rocket_ratio'], df['time_years'], 'o-', color='navy', label='Mean Duration')
    plt.fill_between(df['rocket_ratio'], 
                     df['time_years'] - df['time_std'], 
                     df['time_years'] + df['time_std'], 
                     color='navy', alpha=0.2, label='1-Sigma Uncertainty')
    
    # --- 找回箭头逻辑 ---
    # 找到时间最短的索引（考虑反馈后的实际最优比例）
    opt_idx = df['time_years'].idxmin()
    opt_r = df.loc[opt_idx, 'rocket_ratio']
    opt_t = df.loc[opt_idx, 'time_years']

    plt.annotate(f'Optimal Efficiency Point\n(r={opt_r:.2f}, t={opt_t:.1f}y)', 
                 xy=(opt_r, opt_t), 
                 xytext=(opt_r + 0.1, opt_t + 5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    plt.xlabel('Rocket Usage Ratio ($r$)')
    plt.ylabel('Total Project Duration (Years)')
    plt.title('A. Temporal Impact of Efficiency Feedback', fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot_A_temporal.png', dpi=300)
    plt.close()

    # ==========================================
    # Plot B: Cost-Climate-Risk
    # ==========================================
    plt.figure(figsize=(10, 7))
    sc2 = plt.scatter(df['cost_trillion'], df['temp_rise'], 
                      c=df['risk_score'], s=df['time_years']*5, 
                      cmap='plasma', edgecolors='black', alpha=0.7)
    plt.xlabel('Total Cost (Trillion USD)')
    plt.ylabel('Global Temp Rise (K)')
    plt.title('B. Cost-Climate-Risk Surface (Size=Duration)', fontweight='bold')
    plt.colorbar(sc2, label='System Risk Score')
    plt.grid(True)
    plt.savefig('plot_B_risk_surface.png', dpi=300)
    plt.close()

    # ==========================================
    # Plot C: Multi-Physics Feedback (Core Improvement)
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(10, 7))
    x_plot = np.linspace(0, 1, 100)
    
    # 计算各项物理指标
    eta_vals = [model.calculate_dynamic_eta(x * 5.0) for x in x_plot]
    vis_vals = [model.calculate_visibility(x * 100) for x in x_plot]
    aod_vals = [model.calculate_bc_pollution_index(x * 100) for x in x_plot]
    strat_vals = [model.calculate_stratospheric_disturbance(x * 5.0) for x in x_plot]

    # 左轴: 效率与可见度
    lns1 = ax1.plot(x_plot, eta_vals, 'r-', lw=3, label='Efficiency $\eta$')
    lns2 = ax1.plot(x_plot, np.array(vis_vals)/max(vis_vals), 'm--', lw=2, label='Norm. Visibility')
    ax1.set_xlabel('Rocket Usage Ratio ($r$)')
    ax1.set_ylabel('Performance Metrics ($\eta$, Vis)', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_ylim(0, 1.1)

    # 右轴: AOD 与扰动
    ax2 = ax1.twinx()
    lns3 = ax2.plot(x_plot, aod_vals, 'g-', lw=2, label='AOD (Pollution)')
    lns4 = ax2.plot(x_plot, strat_vals, 'b-.', lw=2, label='Strat. Disturbance')
    ax2.set_ylabel('Atmospheric Disturbance (AOD, Intensity)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    # 合并图例
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right')

    plt.title('C. Integrated Atmospheric Feedback & Efficiency Decay', fontweight='bold')
    ax1.grid(True)
    plt.savefig('plot_C_feedback_physics.png', dpi=300)
    plt.close()

    # ==========================================
    # Plot D: Strategic Comparison (Viridis, Optimized Labels)
    # ==========================================
    plt.figure(figsize=(10, 7))
    
    # 1. 动态获取图 A 中的最优 r 值并取 2 位小数
    opt_idx = df['time_years'].idxmin()
    opt_r_val = df.loc[opt_idx, 'rocket_ratio']
    
    # 2. 定义四个对比点：0, 最优, 0.5, 1.0
    compare_r = [0.0, 0.2, 0.5, 1.0]
    compare_indices = [(df['rocket_ratio'] - r).abs().idxmin() for r in compare_r]
    compare_df = df.iloc[compare_indices].copy()
    
    # 3. 核心修复：格式化标签，解决无限小数问题，并标记 Optimum
    def format_label(r):
        if abs(r - opt_r_val) < 1e-5:
            return f"Optimum ({r:.2f})"
        return f"r={r:.1f}"

    compare_df['Strategy'] = compare_df['rocket_ratio'].apply(format_label)
    
    # 4. 转换数据格式用于 Seaborn
    metrics = ['time_years', 'cost_trillion', 'co2e_million_tons', 'risk_score']
    melted = compare_df.melt(id_vars='Strategy', value_vars=metrics)
    
    # 5. 归一化处理 (基于全局最大值)
    for m in metrics:
        max_v = df[m].max()
        if max_v > 0:
            melted.loc[melted['variable'] == m, 'value'] /= max_v

    # 6. 绘图：使用之前指定的 viridis 颜色
    sns.barplot(data=melted, x='variable', y='value', hue='Strategy', palette='magma')
    
    plt.title('D. Normalized Strategic Comparison (Dynamic Optimal)', fontweight='bold')
    plt.ylabel('Normalized Score (Lower is Better)')
    plt.xticks(ticks=[0, 1, 2, 3], labels=['Speed', 'Cost', 'CO2e', 'Risk'])
    
    # 7. 细节优化：调整图例位置防止遮挡
    plt.legend(title='Strategies', loc='upper right', fontsize='small')
    plt.ylim(0, 1.4) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig('plot_D_comparison.png', dpi=300)
    plt.close()

    print(f"✨ Plot D saved with Viridis palette. Optimum: r={opt_r_val:.2f}")

# 在 main 函数末尾调用
if __name__ == "__main__":
    results_df = run_stochastic_study(n_points=30)
    p = EnvironmentalParameters()
    model = EnvironmentalImpactModel(p)
    save_individual_plots(results_df, model)