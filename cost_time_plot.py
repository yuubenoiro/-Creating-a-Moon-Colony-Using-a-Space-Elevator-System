# optimize.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from model import moon_colony_sim, rocket_only_sim

# ==================================================
# 0. Matplotlib 全局风格（论文友好）
# ==================================================
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

# ==================================================
# 1. 搜索空间
# ==================================================
bounds = [
    (0.0, 0.0),   # alpha（锁死为 0）
    (0.0, 1.0)    # 火箭利用率
]

# ==================================================
# 2. 基准方案
# ==================================================
print(">>> 正在进行：只用火箭")
t0, cost0 = rocket_only_sim(mode='both')

print(">>> 正在进行：只用电梯")
t_ele, cost_ele = moon_colony_sim([0, 0.0], mode='both')

# ==================================================
# 3. 逐步收紧时间约束 → 最小成本
# ==================================================
TIME_LIMITS = np.arange(140, 190, 2)  # 60 → 180 年
best_costs = []
best_times = []

print("\n>>> 时间约束扫描中...")

for T in TIME_LIMITS:
    res = differential_evolution(
        moon_colony_sim,
        bounds,
        args=('cost', None, T),
        strategy='best1bin',
        tol=1e-3,
        polish=True
    )
    t_opt, cost_opt = moon_colony_sim(res.x, mode='both')

    best_times.append(t_opt)
    best_costs.append(cost_opt)

    print(f"  时间 ≤ {T:3d} 年 → 成本 = {cost_opt/1e12:.2f} Trillion")

best_costs = np.array(best_costs) / 1e12  # Trillion USD

# ==================================================
# 4. 绘图：时间约束 vs 最小成本
# ==================================================
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(
    TIME_LIMITS,
    best_costs,
    marker='o',
    linewidth=2.2,
    markersize=5
)

ax.set_xlabel("Time Constraint (years)")
ax.set_ylabel("Minimum Total Cost (Trillion USD)")
ax.set_title("Minimum Cost under Progressive Time Constraints")

ax.grid(True)

# 参考线（只用火箭 & 只用电梯）
ax.axhline(
    cost0 / 1e12,
    linestyle=":",
    linewidth=1.5,
    label="Rocket-only baseline"
)
ax.axhline(
    cost_ele / 1e12,
    linestyle="--",
    linewidth=1.5,
    label="Elevator-only baseline"
)

ax.legend(frameon=False)

# ==================================================
# 5. 保存图片
# ==================================================
plt.tight_layout()
plt.savefig("min_cost_vs_time_constraint.png")
plt.show()

print("\n>>> 图像已保存：min_cost_vs_time_constraint.png")
