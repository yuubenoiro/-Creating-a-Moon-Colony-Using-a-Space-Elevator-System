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
               labels=['Time↓', 'Cost↓', 'CO2e↓', 'Risk↓'])
    
    # 7. 细节优化
    plt.legend(title='Strategies', loc='upper right', fontsize='small')
    plt.ylim(0, 1.4) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig('plot_D_comparison.png', dpi=300)
    plt.close()

    print(f"✨ Plot D saved. Low-impact ratio: r={rec_r_val:.2f}")


# 在 main 函数末尾调用
if __name__ == "__main__":
    results_df = run_stochastic_study(n_points=100)
    p = EnvironmentalParameters()
    model = EnvironmentalImpactModel(p)
    save_individual_plots(results_df, model)