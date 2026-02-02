import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
# 确保 model.py 在同路径下
from model import (
    moon_colony_sim, rocket_only_sim, 
    C_ROCKET_TOTAL, C_ELE_TRANSFER, BASE_COST, TARGET_MASS, MAX_ROCKET_ANNUAL
)

# 1. 运行优化逻辑 (继承自你的 optimize.py)
bounds = [(0.0, 0.5), (0.0, 1.0)]
TIME_LIMIT = 170
BUDGET_LIMIT = 15e12

print(">>> 正在运行优化算法...")
res_c = differential_evolution(moon_colony_sim, bounds, args=('cost', None, TIME_LIMIT))
res_t = differential_evolution(moon_colony_sim, bounds, args=('time', BUDGET_LIMIT))

# 2. 定义绘图函数
def plot_comprehensive_costs():
    plt.style.use('seaborn-v0_8-muted')
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # --- 图 1: 火箭单位成本拆解 (环境 vs 运营) ---
    # 基于 model.py: 400k = 350k(运营) + 50k(污染)
    r_op = 350000
    r_tax = 50000
    axes[0].pie([r_op, r_tax], labels=['Operation Cost', 'Environmental Tax'], 
                autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'], explode=(0, 0.1))
    axes[0].set_title("A. Rocket Unit Cost Breakdown", fontsize=14, fontweight='bold')

    # --- 图 2: 混合优化方案下的总成本构成 ---
    # 提取成本优先方案 (res_c) 的具体构成
    t_fin, c_total = moon_colony_sim(res_c.x, mode='both')
    alpha_opt, r_usage_opt = res_c.x
    
    # 估算混合方案中三者的占比
    total_rocket_mass = MAX_ROCKET_ANNUAL * r_usage_opt * t_fin
    total_ele_mass = TARGET_MASS - total_rocket_mass
    
    cost_from_rocket = total_rocket_mass * C_ROCKET_TOTAL
    cost_from_ele_ops = total_ele_mass * C_ELE_TRANSFER
    cost_infrastructure = BASE_COST # 假设自举成本BS=0暂不计入复杂升级
    
    axes[1].pie([cost_from_rocket, cost_from_ele_ops, cost_infrastructure], 
                labels=['Rocket Ops', 'Elevator Ops', 'Fixed Infrastructure'],
                autopct='%1.1f%%', startangle=90, colors=['#ff6666', '#99ff99', '#ffcc99'],
                explode=(0.05, 0, 0))
    axes[1].set_title(f"B. Hybrid Scenario Cost Mix\n(Time={t_fin} yrs)", fontsize=14, fontweight='bold')

    # --- 图 3: 三大方案总成本对比 (Trillion USD) ---
    _, c_pure_rocket = rocket_only_sim(mode='both')
    _, c_pure_ele = moon_colony_sim([0, 0], mode='both')
    
    labels = ['Pure Rocket', 'Pure Elevator', 'Optimized Hybrid']
    total_costs = [c_pure_rocket/1e12, c_pure_ele/1e12, c_total/1e12]
    
    bars = axes[2].bar(labels, total_costs, color=['#e74c3c', '#2ecc71', '#3498db'])
    axes[2].set_ylabel("Total Cost (Trillion USD)")
    axes[2].set_title("C. Scenario Comparison: Total Expense", fontsize=14, fontweight='bold')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.5, f'${height:.2f}T', ha='center')

    plt.tight_layout()
    plt.savefig("Analysis_Results.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_comprehensive_costs()