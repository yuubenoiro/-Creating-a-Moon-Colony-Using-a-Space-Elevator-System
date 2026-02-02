import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model import moon_colony_sim, rocket_only_sim, TARGET_MASS
from scipy.optimize import differential_evolution

# ===============================
# 1. 数据获取逻辑 (基于你的优化逻辑)
# ===============================
bounds = [(0.0, 0.5), (0.0, 1.0)]
TIME_LIMIT = 170
BUDGET_LIMIT = 15e12

print(">>> 正在运行模型获取数据...")
# Scenario A: 纯电梯
t_a, c_a = moon_colony_sim([0, 0.0], mode='both')
# Scenario B: 纯火箭
t_b, c_b = rocket_only_sim(mode='both')
# Scenario C-1: 成本优先 (混合)
res_c = differential_evolution(moon_colony_sim, bounds, args=('cost', None, TIME_LIMIT))
t_c1, c_c1 = moon_colony_sim(res_c.x, mode='both')
# Scenario C-2: 时间优先 (混合)
res_t = differential_evolution(moon_colony_sim, bounds, args=('time', BUDGET_LIMIT))
t_c2, c_c2 = moon_colony_sim(res_t.x, mode='both')
# Scenario C-3: 时间极小化 (不计成本)
res_min_t = differential_evolution(moon_colony_sim, bounds, args=('time', 1e20)) # 无预算限制
t_c3, c_c3 = moon_colony_sim(res_min_t.x, mode='both')

# ===============================
# 2. 构建美化表格
# ===============================
data = [
    ["A. Space Elevator Only", "0.00%", int(t_a), f"{c_a/1e12:.2f}"],
    ["B. Rockets Only", "100.00%", int(t_b), f"{c_b/1e12:.2f}"],
    ["C-1. Hybrid (Cost Opt)", f"{res_c.x[1]*100:.2f}%", int(t_c1), f"{c_c1/1e12:.2f}"],
    ["C-2. Hybrid (Time Opt)", f"{res_t.x[1]*100:.2f}%", int(t_c2), f"{c_c2/1e12:.2f}"],
    ["C-3. Hybrid (Time Min)", f"{100:.2f}%", int(t_c3), f"{c_c3/1e12:.2f}"]
]

columns = ["Scenario Strategy", "Rocket Usage", "Time (Yrs)", "Total Cost (T USD)"]

def render_mpl_table(data, col_labels, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # 绘制表格
    table = ax.table(cellText=data, colLabels=col_labels, loc='center', cellLoc='center')
    
    # 样式设置
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.2) # 拉伸行高

    # 颜色美化
    for i, key in enumerate(table.get_celld()):
        cell = table.get_celld()[key]
        if key[0] == 0: # 表头
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        elif key[0] == 3: # C-1 推荐行加深颜色
            cell.set_facecolor('#ecf0f1')
            
    plt.savefig("Summary_Table.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    render_mpl_table(data, columns, "Table 1: Multi-Scenario Logistics Optimization Results")
    print(">>> 漂亮表格已保存为 'Summary_Table.png'")