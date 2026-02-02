# model.py
import numpy as np

# ========================
# 1. 全局参数（物理 & 经济）
# ========================

TARGET_MASS = 100_000_000      # 1亿吨 [cite: 11]
INITIAL_CAP = 537_000          # 初始运力 (17.9万吨/年 * 3) [cite: 14, 76]
BASE_COST = 30_000_000_000     # 3条港口, 6条缆绳 * 10B

# 成本参数 (每吨)
C_ROCKET_TOTAL = 400_000       # 350k运费 + 50k污染税
C_ELE_TRANSFER = 105_000       # 仅轨道转移段 (30% of 350k)
C_ELE_UPGRADE_UNIT = 60_000    # 边际扩容单价

# 其他限制
MAX_ROCKET_ANNUAL = 125 * 1188 # 10个基地, 50次/年, 150吨/次 [cite: 16, 19]
BS = 0

def moon_colony_sim(params, mode='cost', budget_limit=None , time_limit=None):
    alpha, r_usage = params
    t, m_accum, total_cost = 0, 0, BASE_COST
    cap_se = INITIAL_CAP
    
    # 模拟 2050 开始的 150 年
    while m_accum < TARGET_MASS and t < 1000:
        # 1. 运力分配
        # 纯火箭：直接从地球运往月球
        m_rocket = MAX_ROCKET_ANNUAL * r_usage
        
        # 电梯：alpha用于自举升级，(1-alpha)用于送往月球
        m_ele_to_moon = cap_se * (1 - alpha)
        m_ele_to_upgrade = cap_se * alpha
        
        # 2. 费用计算
        cost_rocket = m_rocket * C_ROCKET_TOTAL
        cost_ele_path = m_ele_to_moon * C_ELE_TRANSFER
        
        # 升级费用：扩容吨数 * 6万/吨/年
        # 假设 1 吨留在 GEO 的材料可转化为 0.2 吨/年的永久运力 (可调效率)
        new_cap = m_ele_to_upgrade * BS
        cost_upgrade = new_cap * C_ELE_UPGRADE_UNIT     
        
        # 3. 累计与更新
        m_accum += (m_rocket + m_ele_to_moon)
        total_cost += (cost_rocket + cost_ele_path + cost_upgrade)
        cap_se += new_cap
        t += 1
        
    if mode == 'cost': 
        if time_limit and t > time_limit:
            return total_cost + (t - time_limit) * 1e20
        return total_cost
    if mode == 'time':
        # 预算惩罚：如果超支，给完工时间加一个极大的惩罚项
        if budget_limit and total_cost > budget_limit:
            return t + (total_cost - budget_limit) * 1e20
        return t
    return t, total_cost

def rocket_only_sim(mode='both'):
    """
    只使用火箭运输，计算完成 TARGET_MASS 所需时间和成本
    mode:
        'time' : 返回完成时间
        'cost' : 返回总成本
        'both' : 返回 (时间, 成本)
    """
    t = 0
    m_accum = 0
    total_cost = BASE_COST  # 初始固定成本

    while m_accum < TARGET_MASS and t < 1000:
        m_rocket = MAX_ROCKET_ANNUAL
        m_accum += m_rocket
        total_cost += m_rocket * C_ROCKET_TOTAL
        t += 1

    if mode == 'time':
        return t
    elif mode == 'cost':
        return total_cost
    else:  # both
        return t, total_cost
