import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from model import moon_colony_sim

# ==================================================
# 1. Matplotlib 风格（学术向）
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
# 2. 搜索空间
# ==================================================
bounds = [
    (0.0, 0.0),   # alpha 固定
    (0.0, 1.0)    # 火箭利用率
]

# ==================================================
# 3. 扫描时间约束 → Pareto 点
# ==================================================
TIME_LIMITS = np.arange(60, 181, 10)

pareto_time = []
pareto_cost = []

print(">>> 计算 Pareto Frontier 中...")

for T_max in TIME_LIMITS:
    res = differential_evolution(
        moon_colony_sim,
        bounds,
        args=('cost', None, T_max),
        strategy='best1bin',
        tol=1e-3,
        polish=True
    )

    T_real, C_real = moon_colony_sim(res.x, mode='both')

    pareto_time.append(T_real)
    pareto_cost.append(C_real / 1e12)  # Trillion USD

    print(f"  T ≤ {T_max:3d} 年 → (T, C) = ({T_real:.1f}, {C_real/1e12:.2f})")

pareto_time = np.array(pareto_time)
pareto_cost = np.array(pareto_cost)

# ==================================================
# 4. Pareto Frontier 绘图
# ==================================================
fig, ax = plt.subplots(figsize=(7, 4.8))

ax.plot(
    pareto_time,
    pareto_cost,
    marker='o',
    linewidth=2.2,
    markersize=5,
    label="Pareto Frontier"
)

ax.set_xlabel("Completion Time (years)")
ax.set_ylabel("Total Cost (Trillion USD)")
ax.set_title("Time–Cost Pareto Frontier for Lunar Logistics System")

ax.grid(True)
ax.legend(frameon=False)

# ==================================================
# 5. 保存图像
# ==================================================
plt.tight_layout()
plt.savefig("pareto_frontier_time_cost.png")
plt.show()

print("\n>>> Pareto Frontier 已保存：pareto_frontier_time_cost.png")
