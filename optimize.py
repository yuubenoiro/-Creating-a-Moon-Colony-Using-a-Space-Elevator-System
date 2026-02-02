# optimize.py
from scipy.optimize import differential_evolution
from model import moon_colony_sim, rocket_only_sim

# ===============================
# 1. 搜索空间
# ===============================

bounds = [
    (0.0, 0.0),   # alpha：0–50% 用于自举
    (0.0, 1.0)    # r_usage：火箭利用率
]

# ===============================
# 2. 优化目标 1：最小总成本
# ===============================

print(">>> 正在进行：只用火箭")

t0, cost0 = rocket_only_sim(mode='both')
print(f"只用火箭完成时间: {t0} 年")
print(f"只用火箭总成本: ${cost0/1e12:.2f} Trillion")

print("\n>>> 正在进行：只用电梯（alpha=0, r_usage=0）")

t_ele, cost_ele = moon_colony_sim([0, 0.0], mode='both')
print(f"只用电梯完成时间: {t_ele} 年")
print(f"只用电梯总成本: ${cost_ele/1e12:.2f} Trillion")

# ===============================
# 3. 优化目标 1：最小总成本
# ===============================

TIME = 170  # 时间上限

res_c = differential_evolution(
    moon_colony_sim,
    bounds,
    args=('cost', None, TIME),
    strategy='best1bin',
    tol=1e-3
)
t1, cost1 = moon_colony_sim(res_c.x, mode='both')

# ===============================
# 4. 优化目标 2：预算约束下最短时间
# ===============================

BUDGET = 15_000_000_000_000  # 12 Trillion

res_t = differential_evolution(
    moon_colony_sim,
    bounds,
    args=('time', BUDGET),
    strategy='best1bin',
    tol=1e-3
)
t2, cost2 = moon_colony_sim(res_t.x, mode='both')

# ===============================
# 5. 输出结果
# ===============================

print("\n--- 优化结果（基于最新成本表）---")

print(f"①  成本优先策略（时间170年）：")
print(f"   Alpha = {res_c.x[0]:.2%}")
print(f"   火箭利用率 = {res_c.x[1]:.2%}")
print(f"   用时 = {t1} 年")
print(f"   总成本 = ${cost1/1e12:.2f} Trillion")

print(f"\n② 时间优先策略（预算 {BUDGET/1e12:.2f} Trillion）：")
print(f"   Alpha = {res_t.x[0]:.2%}")
print(f"   火箭利用率 = {res_t.x[1]:.2%}")
print(f"   用时 = {t2} 年")
print(f"   总成本 = ${cost2/1e12:.2f} Trillion")