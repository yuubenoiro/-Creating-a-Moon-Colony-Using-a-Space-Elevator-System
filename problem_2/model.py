import numpy as np
import matplotlib.pyplot as plt

# ========================
# 1. 固定参数 (基于你的 model.py)
# ========================
TARGET_MASS = 100_000_000      
INITIAL_CAP = 537_000          
BASE_COST = 30_000_000_000     
C_ROCKET_TOTAL = 400_000       
C_ELE_TRANSFER = 105_000       
C_ELE_UPGRADE_UNIT = 60_000    
MAX_ROCKET_ANNUAL = 125 * 1188 
BS = 0.2  # 建议将BS调回非0值（如0.2），否则无法体现电梯升级的抗风险能力

# ========================
# 2. 第二问新增：风险参数
# ========================
ETA = 0.85                   # 摆动导致的运力打折系数
ROCKET_SUCCESS_RATE = 0.95   # 火箭成功率 (均匀分布判断)
ELE_FAULT_LAMBDA = 20      # 电梯年均故障率 (泊松分布λ)
DOWNTIME_PER_FAULT = 0.02     # 每次电梯故障导致当年停工 20% 的时间
ELE_BIG_FAULT_LAMBDA = 0.3
DOWNTIME_PER_BIG_FAULT = 0.5

def stochastic_sim(params):
    """单次蒙特卡洛模拟过程"""
    alpha, r_usage = params
    t, m_accum, total_cost = 0, 0, BASE_COST
    cap_se = INITIAL_CAP
    
    while m_accum < TARGET_MASS and t < 1000:
        # --- 火箭部分 (均匀分布模拟故障) ---
        # 假设年发射总量由多次小规模任务组成，成功量符合二项分布(由均匀分布累加得出)
        planned_rocket_mass = MAX_ROCKET_ANNUAL * r_usage
        # 实际运抵量 = 计划量 * 随机成功因子
        actual_rocket_mass = np.random.binomial(n=100, p=ROCKET_SUCCESS_RATE) / 100 * planned_rocket_mass
        
        # --- 电梯部分 (泊松分布模拟重大故障) ---
        # 产生这一年的故障次数
        num_faults = np.random.poisson(ELE_FAULT_LAMBDA)
        num_big_faults = np.random.poisson(ELE_BIG_FAULT_LAMBDA)
        # 实际可用时间因子 (考虑故障停机)
        availability = max(0.1, 1.0 - (num_faults * DOWNTIME_PER_FAULT) - (num_big_faults * DOWNTIME_PER_BIG_FAULT))
        
        # 电梯运力受 ETA 和 故障停机 双重影响
        effective_cap = cap_se * ETA * availability
        
        m_ele_to_moon = effective_cap * (1 - alpha)
        m_ele_to_upgrade = effective_cap * alpha
        
        # --- 费用核算 (失败的火箭也要算钱) ---
        cost_rocket = planned_rocket_mass * C_ROCKET_TOTAL 
        cost_ele_path = m_ele_to_moon * C_ELE_TRANSFER
        new_cap = m_ele_to_upgrade * BS
        cost_upgrade = new_cap * C_ELE_UPGRADE_UNIT
        
        # --- 更新状态 ---
        m_accum += (actual_rocket_mass + m_ele_to_moon)
        total_cost += (cost_rocket + cost_ele_path + cost_upgrade)
        cap_se += new_cap
        t += 1
        
    return t, total_cost

# ========================
# 3. 蒙特卡洛执行环境
# ========================
def run_monte_carlo(params, iterations=1000):
    results = [stochastic_sim(params) for _ in range(iterations)]
    t_list, c_list = zip(*results)
    
    print(f"--- 蒙特卡洛模拟结果 ({iterations} 次运行) ---")
    print(f"参数：η = {ETA}   λ_小故障 = {ELE_FAULT_LAMBDA}  downtime_小故障 = {DOWNTIME_PER_FAULT}年   λ_大故障 = {ELE_BIG_FAULT_LAMBDA}  downtime_大故障 = {DOWNTIME_PER_BIG_FAULT}年   火箭成功率 = {ROCKET_SUCCESS_RATE}")
    print(f"平均完工时间: {np.mean(t_list):.2f} 年 (标准差: {np.std(t_list):.2f})")
    print(f"平均总成本: ${np.mean(c_list)/1e12:.2f} 万亿 (标准差: {np.std(c_list)/1e12:.2f})")
    print(f"95% 置信区间 (时间): [{np.percentile(t_list, 2.5):.0f}, {np.percentile(t_list, 97.5):.0f}] 年")
    
    return t_list, c_list

# 测试你的折中方案 (例如 alpha=0.15, r_usage=0.4)
p_balanced = [0, 0.4]
t_dist, c_dist = run_monte_carlo(p_balanced)