import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from constants import EnvironmentalParameters
from engines import EnvironmentalImpactModel
from analyzer5 import DecisionAnalyzerV4 as DecisionAnalyzerV2
from scipy.integrate import odeint

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
    # Plot B: Geometric Curvature Optimization (Knee Point)
    # ==========================================
    plt.figure(figsize=(10, 7))
    
    # 1. 准备三维空间向量并进行标准化 (微电子中常用的信号完整性分析方法)
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min())

    t_n = normalize(df['time_years']).values
    r_n = normalize(df['risk_score']).values
    c_n = normalize(df['cost_trillion']).values
    points = np.vstack((t_n, r_n * 20, c_n)).T

    # 2. 计算离散点集的曲率 (Menger Curvature)
    curvatures = [0] # 首尾点曲率设为0
    for i in range(1, len(points) - 1):
        p1, p2, p3 = points[i-1], points[i], points[i+1]
        # 计算三角形面积 (海伦公式)
        a = np.linalg.norm(p1 - p2)
        b = np.linalg.norm(p2 - p3)
        c = np.linalg.norm(p3 - p1)
        s = (a + b + c) / 2
        area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
        kappa = (4 * area) / (a * b * c + 1e-6)
        curvatures.append(kappa)
    curvatures.append(0)
    df['curvature'] = curvatures

    # 3. 寻找曲率最大的点 (即性能折点)
    # 我们通常在策略的中段（r=0.2~0.8）寻找拐点，排除两端极端情况
    mask = (df['rocket_ratio'] > 0.05) & (df['rocket_ratio'] < 0.95)
    best_idx = df[mask]['curvature'].idxmax()
    
    knee_t = df.loc[best_idx, 'time_years']
    knee_r = df.loc[best_idx, 'risk_score']
    knee_c = df.loc[best_idx, 'cost_trillion']
    knee_ratio = df.loc[best_idx, 'rocket_ratio']

    # 4. 绘图：Time vs Risk，颜色映射 Cost
    sc = plt.scatter(df['time_years'], df['risk_score'], 
                     c=df['cost_trillion'], cmap='YlOrRd', 
                     s=100, edgecolors='black', alpha=0.7, zorder=2)
    
    # 5. 标注曲率最优点
    plt.scatter(knee_t, knee_r, color='lime', marker='s', s=200, 
                edgecolors='black', label='Geometric Knee Point (Opt)', zorder=3)
    
    plt.annotate(f'Knee Point Optimization\nMax Curvature at r={knee_ratio:.2f}\nCost=${knee_c:.1f}T', 
                 xy=(knee_t, knee_r), xytext=(knee_t + 15, knee_r + 10),
                 arrowprops=dict(arrowstyle='fancy', connectionstyle="arc3,rad=.2", fc="lime"),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round", fc="lime", alpha=0.2))

    plt.xlabel('Project Duration (Time, Years)')
    plt.ylabel('System Risk Score')
    plt.title('B. Pareto Frontier Analysis via Curvature Maximization', fontweight='bold')
    plt.colorbar(sc, label='Total Cost (Trillion USD)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plt.savefig('plot_B_optimization.png', dpi=300)
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
    
    # 1. 动态获取推荐的 r 值（不再称之为 Optimum）
    rec_idx = df['time_years'].idxmin()
    rec_r_val = df.loc[rec_idx, 'rocket_ratio']
    
    # 2. 定义对比点
    compare_r = [0.0, 0.2, 0.5, 1.0]
    compare_indices = [(df['rocket_ratio'] - r).abs().idxmin() for r in compare_r]
    compare_df = df.iloc[compare_indices].copy()
    
    # 3. 修改标签逻辑：将 "Optimum" 改为 "Low-impact"
    def format_label(r):
        if abs(r - 0.2) < 1e-5: # 对应你论文中的 r=0.2 
            return f"Low-impact ({r:.1f})"
        return f"r={r:.1f}"

    compare_df['Strategy'] = compare_df['rocket_ratio'].apply(format_label)
    
    # 4. 数据转换与归一化 (保持不变)
    metrics = ['time_years', 'cost_trillion', 'co2e_million_tons', 'risk_score']
    melted = compare_df.melt(id_vars='Strategy', value_vars=metrics)
    
    for m in metrics:
        max_v = df[m].max()
        if max_v > 0:
            melted.loc[melted['variable'] == m, 'value'] /= max_v

    # 5. 绘图
    # 注意：为了让图表更有科技感，建议将 speed 放在第一位
    sns.barplot(data=melted, x='variable', y='value', hue='Strategy', palette='magma')
    
    plt.title('D. Normalized Strategic Comparison (Environmental Performance)', fontweight='bold')
    plt.ylabel('Normalized Score')

    # 6. 核心修改：在 X 轴标签中添加箭头 (↑ 表示更高越好，↓ 表示越低越好)
    # 注意：原本的 'time_years' 越小越好，所以标记为 ↓ ；如果你想表达“速度”，则标记为 ↑
    # 这里我们统一使用你要求的方向：Speed ↑, 其他 ↓
    plt.xticks(ticks=[0, 1, 2, 3], 
               labels=['Speed↑', 'Cost↓', 'CO2e↓', 'Risk↓'])
    
    # 7. 细节优化
    plt.legend(title='Strategies', loc='upper right', fontsize='small')
    plt.ylim(0, 1.4) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig('plot_D_comparison.png', dpi=300)
    plt.close()

    print(f"✨ Plot D saved. Low-impact ratio: r={rec_r_val:.2f}")

    # ==========================================
    # Plot E: Temperature Rise Projection (Strictly Positive Y-axis)
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    t_span = np.linspace(0, 20, 200) 
    plot_r_list = [0.0, 0.2, 1.0]
    colors = ['#1a9641', '#0571b0', '#ca0020'] 
    
    max_temp = 0 # 用于动态调整坐标轴上限
    
    for r_val, color in zip(plot_r_list, colors):
        avg_emission_rate = (r_val * 5.0) 
        initial_state = [0.0, 0.0, 0.0]
        
        # 求解 ODE
        sol = odeint(model._bc_climate_ode, initial_state, t_span, args=(avg_emission_rate,))
        temp_rise = sol[:, 2]
        max_temp = max(max_temp, temp_rise.max())
        
        if r_val == 0.0:
            label = "Pure Elevator (r=0.0)"
        elif r_val == 1.0:
            label = "Aggressive Rocketry (r=1.0)"
        else:
            label = f"Hybrid Strategy (r={r_val:.2f})"
            
        plt.plot(t_span, temp_rise, label=label, color=color, lw=3, zorder=3)

    # 装饰美化
    plt.title('E. Projected Global Temperature Rise (20-Year Horizon)', fontsize=14, fontweight='bold')
    plt.xlabel('Years from Project Start', fontsize=12)
    plt.ylabel('Temperature Change $\Delta T$ (°C)', fontsize=12)
    
    # --- 核心修改：锁定纵轴范围，去除负数部分 ---
    plt.ylim(0, max_temp * 1.15) 
    plt.xlim(0, 20)
    
    
    plt.grid(True, linestyle='--', alpha=0.4, zorder=1)
    plt.legend(loc='upper left', frameon=True, shadow=False)
    
    plt.tight_layout()
    plt.savefig('plot_E_temperature.png', dpi=300)
    plt.close()

    print(f"✨ Plot E (Positive Y-axis) saved. Max Delta T: {max_temp:.4f}")

# 在 main 函数末尾调用
if __name__ == "__main__":
    results_df = run_stochastic_study(n_points=100)
    p = EnvironmentalParameters()
    model = EnvironmentalImpactModel(p)
    save_individual_plots(results_df, model)