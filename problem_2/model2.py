import numpy as np

# ========================
# 1. 固定参数
# ========================
TARGET_MASS = 100_000_000
INITIAL_CAP = 537_000
BASE_COST = 30_000_000_000
C_ROCKET_TOTAL = 400_000
C_ELE_TRANSFER = 105_000
C_ELE_UPGRADE_UNIT = 60_000
BS = 0.2 

# 火箭二项分布参数 [根据你的要求修改]
N_ROCKET_WINDOW = 1188      # 每年最大发射次数 n
P_ROCKET_SUCCESS = 0.95     # 单次发射成功率 p
ROCKET_LOAD = 125           # 每次发射的载荷 (1188 * 125 = 原 MAX_ROCKET_ANNUAL)

# ========================
# 2. 动态故障参数 (大小故障结合)
# ========================
ETA = 0.85
# 小故障：时间衰减
LAMBDA_SMALL_INIT = 20.0
DECAY_RATE = 0.05
DOWNTIME_SMALL = 0.02
# 大故障：固定泊松
LAMBDA_BIG_FIXED = 0.3
DOWNTIME_BIG = 0.5

def stochastic_sim_v4(params):
    alpha, r_usage = params # r_usage 现在决定了实际动用的发射窗口比例 (0-1)
    t, m_accum, total_cost = 0, 0, BASE_COST
    cap_se = INITIAL_CAP
    
    while m_accum < TARGET_MASS and t < 1000:
        # --- A. 火箭部分：二项分布 B(n, p) ---
        # 实际动用的窗口数 n_active
        n_active = int(N_ROCKET_WINDOW * r_usage)
        # 成功发射次数符合二项分布
        success_launches = np.random.binomial(n=n_active, p=P_ROCKET_SUCCESS)
        actual_rocket_mass = success_launches * ROCKET_LOAD
        
        # --- B. 电梯部分：动态可靠性 ---
        curr_lambda_small = LAMBDA_SMALL_INIT * np.exp(-DECAY_RATE * t)
        num_small = np.random.poisson(curr_lambda_small)
        num_big = np.random.poisson(LAMBDA_BIG_FIXED)
        
        # 考虑故障导致的可用性下降
        availability = max(0.1, 1.0 - (num_small * DOWNTIME_SMALL) - (num_big * DOWNTIME_BIG))
        effective_cap = cap_se * ETA * availability
        
        m_ele_to_moon = effective_cap * (1 - alpha)
        m_ele_to_upgrade = effective_cap * alpha
        
        # --- C. 财务与状态更新 ---
        # 成本计算：注意，只要准备发射(n_active)，无论成功失败都要付钱
        total_cost += (n_active * ROCKET_LOAD * C_ROCKET_TOTAL) 
        total_cost += (m_ele_to_moon * C_ELE_TRANSFER)
        
        new_cap = m_ele_to_upgrade * BS
        total_cost += (new_cap * C_ELE_UPGRADE_UNIT)
        
        m_accum += (actual_rocket_mass + m_ele_to_moon)
        cap_se += new_cap
        t += 1
        
    return t, total_cost


import numpy as np

def run_monte_carlo_v4(params, iterations=1000):
    # 执行模拟
    results = [stochastic_sim_v4(params) for _ in range(iterations)]
    t_list, c_list = zip(*results)
    
    # 统计计算
    mean_t, std_t = np.mean(t_list), np.std(t_list)
    mean_c, std_c = np.mean(c_list), np.std(c_list)
    ci_t = [np.percentile(t_list, 2.5), np.percentile(t_list, 97.5)]
    ci_c = [np.percentile(c_list, 2.5), np.percentile(c_list, 97.5)]

    # 打印更详尽的风险评估报告
    print(f"\n" + "="*50)
    print(f"  第二问：不完美运行状态下的蒙特卡洛模拟 (n={iterations})")
    print(f"  模型特性：可靠性增长 (衰减率 {DECAY_RATE*100:.1f}%) + 离散二项分布火箭")
    print(f"  策略参数：Alpha = {params[0]:.2%}, Rocket Usage = {params[1]:.2%}")
    print("="*50)
    
    print(f"【完工时间 (Time)】")
    print(f"  平均值: {mean_t:.2f} 年")
    print(f"  标准差: {std_t:.2f} 年 (体现了系统波动性)")
    print(f"  95% 置信区间: [{ci_t[0]:.0f}, {ci_t[1]:.0f}] 年")
    
    print(f"\n【总成本 (Cost)】")
    print(f"  平均值: ${mean_c/1e12:.2f} 万亿")
    print(f"  标准差: ${std_c/1e12:.2f} 万亿")
    print(f"  95% 置信区间: [${ci_c[0]/1e12:.2f}T, ${ci_c[1]/1e12:.2f}T]")
    
    # 鲁棒性评价
    risk_index = std_t / mean_t
    print(f"\n【鲁棒性评估】")
    print(f"  时间变异系数 (CV): {risk_index:.4f}")
    if risk_index < 0.05:
        print("  评价：该方案鲁棒性极佳，受随机故障影响较小。")
    else:
        print("  评价：该方案存在时间风险，建议增加运力冗余。")
    print("="*50)
    
    return t_list, c_list

# ========================
# 运行模拟
# ========================

# 测试方案 A: alpha=0 (你目前设定的无自举方案)
# 测试方案 B: alpha=0.15 (引入自举的方案，对比两者的抗风险能力)
print("\n正在运行无自举方案 (Alpha=0)...")
t_dist_fixed, c_dist_fixed = run_monte_carlo_v4([0.0, 0.4])

print("\n正在运行自举方案 (Alpha=0.15)...")
t_dist_boost, c_dist_boost = run_monte_carlo_v4([0.15, 0.4])