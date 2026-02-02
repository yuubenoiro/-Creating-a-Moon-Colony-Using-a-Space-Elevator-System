import numpy as np

# ==========================================
# 1. 基础参数 (继承自第二问)
# ==========================================
INITIAL_CAP = 537_000
TARGET_MASS = 100_000_000
BS = 0.2
C_ELE_TRANSFER = 105_000
C_ROCKET_TOTAL = 400_000
ETA = 0.85
LAMBDA_BIG = 0.3
DOWNTIME_BIG = 0.5
DECAY_RATE = 0.05
LAMBDA_SMALL_INIT = 20.0
DOWNTIME_SMALL = 0.02

# ==========================================
# 2. 第三问特定：水资源参数
# ==========================================
POPULATION = 100_000
# 初始库存 (存量): 支撑“lush green”环境
M_INV = POPULATION * 1.5  
# 年度损耗补给 (流量): 基于 98% 回收率
M_MAKEUP = 2555           
M_WATER_TOTAL = M_INV + M_MAKEUP  # 总交付目标: 152,555 吨

# ==========================================
# 3. 阶段 1：基建仿真 (获取 2050+ 年的状态)
# ==========================================
def run_construction_phase(alpha, r_usage):
    m_accum = 0
    cap_se = INITIAL_CAP
    t = 0
    while m_accum < TARGET_MASS:
        # 简化计算：获取建设完成时的理论运力和时间
        # 在实际论文中，这一步可以用你第二问的完整蒙特卡洛均值
        m_step = cap_se * (1 - alpha) * ETA
        m_accum += m_step
        cap_se += (cap_se * alpha * ETA) * BS
        t += 1
    return t, cap_se

# ==========================================
# 4. 阶段 2：第三问 - 水资源任务仿真
# ==========================================
def water_mission_sim(final_cap, t_start, mode='elevator'):
    """
    mode: 'elevator' 或 'rocket'
    """
    m_water_accum = 0
    days = 0
    total_cost = 0
    
    # 转换为日运力进行高精度仿真
    daily_cap_se = (final_cap * ETA) / 365
    daily_cap_rocket = (125 * 1188 * 10) / 365 # 假设10个站点全力补给
    
    while m_water_accum < M_WATER_TOTAL:
        # 故障判定
        num_big = np.random.poisson(LAMBDA_BIG / 365)
        # 小故障已随时间衰减至低位
        curr_lambda_small = LAMBDA_SMALL_INIT * np.exp(-DECAY_RATE * t_start)
        num_small = np.random.poisson(curr_lambda_small / 365)
        
        availability = max(0.1, 1.0 - (num_big * DOWNTIME_BIG) - (num_small * DOWNTIME_SMALL))
        
        if mode == 'elevator':
            m_daily = daily_cap_se * availability
            total_cost += m_daily * C_ELE_TRANSFER
        else:
            m_daily = daily_cap_rocket * 0.95 # 95%成功率
            total_cost += m_daily * C_ROCKET_TOTAL
            
        m_water_accum += m_daily
        days += 1
        
    return days, total_cost

# ==========================================
# 5. 综合评估报告
# ==========================================
def evaluate_phase_3(alpha_val):
    # 1. 预运行基建阶段，获取状态
    t_const, cap_end = run_construction_phase(alpha=alpha_val, r_usage=0.4)
    
    # 2. 运行水资源任务蒙特卡洛 (1000次)
    se_results = [water_mission_sim(cap_end, t_const, 'elevator') for _ in range(1000)]
    rock_results = [water_mission_sim(cap_end, t_const, 'rocket') for _ in range(1000)]
    
    avg_days_se, avg_cost_se = np.mean(se_results, axis=0)
    avg_days_rk, avg_cost_rk = np.mean(rock_results, axis=0)

    print(f"=== 第三问：水资源保障任务结论 (Alpha={alpha_val}) ===")
    print(f"基建结束时电梯运力: {cap_end/1e6:.2f} M Tons/Year")
    print(f"总运输任务量: {M_WATER_TOTAL:,.0f} Tons")
    print("-" * 40)
    print(f"【方案 A: 空间电梯】")
    print(f"  平均耗时: {avg_days_se:.2f} 天")
    print(f"  追加成本: ${avg_cost_se/1e9:.2f} Billion")
    print(f"\n【方案 B: 传统火箭】")
    print(f"  平均耗时: {avg_days_rk:.2f} 天")
    print(f"  追加成本: ${avg_cost_rk/1e9:.2f} Billion")
    print("-" * 40)
    print(f"性价比分析: 电梯方案比火箭节省了 ${(avg_cost_rk - avg_cost_se)/1e9:.2f} Billion")

# 运行评估 (以自举方案为例)
evaluate_phase_3(alpha_val=0.15)