# analyzers.py
import numpy as np
from engines import EnvironmentalImpactModel

class DecisionAnalyzer:
    """多目标决策分析器 - 集成环境、成本、时间与综合风险评估"""
    
    def __init__(self, env_model: EnvironmentalImpactModel):
        self.model = env_model
        self.params = env_model.params

    def calculate_risk_score(self, rocket_usage_ratio: float) -> float:
        """
        计算综合风险评分 (0-100, 越高越不稳定)
        
        逻辑:
        1. 技术可靠性: 基于火箭和电梯各自的故障率加权
        2. 单点故障风险: U型曲线，过度依赖单一系统(r=0或1)风险高
        3. 供应链多样性: 混合运输模式下的鲁棒性奖励
        """
        r = rocket_usage_ratio
        
        # 1. 技术可靠性风险 (基础线性加权)
        reliability_risk = (
            r * self.params.ROCKET_FAILURE_RATE * 100 +
            (1 - r) * self.params.ELEVATOR_FAILURE_RATE * 100
        )
        
        # 2. 单点故障风险 (U型曲线: 当 r=0.5 时风险最小)
        # 这里的 50 是振幅，模拟极端依赖单一系统的系统脆弱性
        single_point_risk = 50 * (1 - 4 * r * (1 - r))
        
        # 3. 供应链多样性奖励 (负值代表风险降低)
        # 当混合程度最高(r=0.5)时，奖励最大(-20)
        diversity_bonus = -20 * r * (1 - r) * 4
        
        total_risk = reliability_risk + single_point_risk + diversity_bonus
        return np.clip(total_risk, 0, 100)

    def evaluate_strategy(self, total_mass, r_usage):
        """
        r_usage: 0.0 到 1.0，代表启用火箭最大发射频率的百分比
        """
        # --- 1. 运力与时间计算 (并发流模型) ---
        cap_ele = self.params.ELEVATOR_CAPACITY_YEAR
        max_cap_rocket = self.params.ANNUAL_ROCKET_LAUNCH * self.params.ROCKET_PAYLOAD
        
        current_cap_rocket = max_cap_rocket * r_usage
        total_annual_capacity = cap_ele + current_cap_rocket
        
        # 总时间 = 总质量 / 总年运力
        time = total_mass / total_annual_capacity

        # --- 2. 任务分配与成本计算 ---
        # 实际运输中，火箭分担的货物比例 (用于成本和排放计算)
        actual_rocket_share = current_cap_rocket / total_annual_capacity
        mass_rocket = actual_rocket_share * total_mass
        mass_ele = (1 - actual_rocket_share) * total_mass
        
        cost = (mass_rocket * self.params.ROCKET_COST_PER_TON + 
                mass_ele * self.params.ELEVATOR_COST_PER_TON) / 1e12 # Trillion $

        # --- 3. 风险评分计算 ---
        # 注意：风险基于“运力依赖度”计算更为合理
        risk = self.calculate_risk_score(actual_rocket_share)

        # --- 4. 环境影响计算 ---
        # 传入实际承载比例进行 ODE 求解
        env = self.model.calculate_hybrid_emissions(total_mass, actual_rocket_share)
        
        return {
            'rocket_ratio': r_usage,           # 火箭启用率
            'actual_mass_share': actual_rocket_share, # 实际载荷分配比
            'time_years': time,
            'cost_trillion': cost,
            'risk_score': risk,
            'temp_rise': env['temp_rise'],
            'co2e_million_tons': env['co2e_total'] / 1e6,
            'bc_tons': total_mass * actual_rocket_share * self.params.ROCKET_BC_FACTOR / 1000, # 吨
            'num_launches': np.ceil(mass_rocket / self.params.ROCKET_PAYLOAD)
        }

    def pareto_analysis(self, total_mass: float, n_points: int = 50):
        """
        帕累托前沿分析
        """
        ratios = np.linspace(0.0, 1.0, n_points)
        return [self.evaluate_strategy(total_mass, r) for r in ratios]