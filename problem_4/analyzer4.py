# analyzer4.py
import numpy as np
from engines import EnvironmentalImpactModel

class DecisionAnalyzerV4:
    """
    环境-安全耦合分析器 (Environment-Security Coupled Analyzer)
    引入黑碳排放对系统风险的非线性惩罚效应。
    """
    def __init__(self, env_model: EnvironmentalImpactModel):
        self.model = env_model
        self.params = env_model.params

    def _single_stochastic_run(self, total_mass, r_usage, alpha=0.0):
        """
        单次随机仿真，集成动态 Eta
        """
        t, m_accum, total_cost = 0, 0, 0
        cap_se = self.params.ELEVATOR_CAPACITY_YEAR
        
        # 记录每年的效率，用于分析
        eff_history = []

        while m_accum < total_mass and t < 1000:
            # --- A. 火箭发射与黑碳映射 ---
            n_active = int(self.params.ANNUAL_ROCKET_LAUNCH * r_usage)
            success_launches = np.random.binomial(n=n_active, p=self.params.P_ROCKET_SUCCESS)
            actual_rocket_mass = success_launches * self.params.ROCKET_LOAD
            
            # --- B. 动态效率反馈 ---
            bc_x = r_usage * 5.0  # 强度映射
            dynamic_eta = self.model.calculate_dynamic_eta(bc_x)
            eff_history.append(dynamic_eta)
            
            # --- C. 电梯随机故障 ---
            # 随时间演化的故障率（可靠性增长）
            p_small = self.params.LAMBDA_SMALL_INIT * np.exp(-self.params.DECAY_RATE * t)
            num_small = np.random.poisson(p_small)
            num_big = np.random.poisson(self.params.LAMBDA_BIG_FIXED)
            
            # 停机时间对可用性的影响
            availability = max(0.1, 1.0 - (num_small * 0.02 + num_big * 0.5))
            
            # 有效运力 = 基础运力 * 动态环境效率 * 随机可用性
            effective_cap = cap_se * dynamic_eta * availability
            
            # --- D. 财务与状态更新 ---
            m_ele_to_moon = effective_cap * (1 - alpha)
            m_accum += (actual_rocket_mass + m_ele_to_moon)
            
            # 成本计算
            total_cost += (n_active * self.params.ROCKET_LOAD * self.params.ROCKET_COST_PER_TON)
            total_cost += (m_ele_to_moon * self.params.ELEVATOR_COST_PER_TON)
            
            # 运力自举
            cap_se += (effective_cap * alpha * self.params.BS)
            t += 1
            
        return t, total_cost / 1e12, np.mean(eff_history) if eff_history else 0

    def evaluate_strategy(self, total_mass, r_usage, alpha=0.0, iterations=50):
        """
        综合评估：集成进度波动风险与黑碳环境风险
        """
        t_list, c_list, eta_list = [], [], []
        
        for _ in range(iterations):
            t, c, e = self._single_stochastic_run(total_mass, r_usage, alpha)
            t_list.append(t)
            c_list.append(c)
            eta_list.append(e)
            
        mean_t = np.mean(t_list)
        std_t = np.std(t_list)
        
        # --- 风险评分模型更新 ---
        
        # 1. 进度稳健性风险 (进度越抖动，分数越高)
        volatility_risk = (std_t / mean_t) * 150 if mean_t > 0 else 0
        
        # 2. 结构性冗余风险 (过度依赖单一方式)
        redundancy_risk = 40 * (1 - 4 * r_usage * (1 - r_usage))
        
        # 3. 黑碳诱发环境风险 (核心新增)
        # 映射 r_usage 到黑碳强度并计算平流层扰动
        bc_x = r_usage * 5.0
        strat_disturb = self.model.calculate_stratospheric_disturbance(bc_x)
        # 扰动指标 y 越大，环境安全性分值越高（惩罚越大）
        # y0=0.1 时基本无惩罚，y 接近 1.3 时产生约 30 分的额外风险
        env_safety_risk = (strat_disturb - 0.1) * 25 
        
        # 综合风险评分：加权组合
        total_risk = np.clip(volatility_risk * 0.4 + redundancy_risk * 0.3 + env_safety_risk * 0.3, 0, 100)
        
        # 环境指标计算
        env_metrics = self.model.calculate_hybrid_emissions(total_mass, r_usage)

        return {
            'rocket_ratio': r_usage,
            'alpha': alpha,
            'time_years': mean_t,
            'time_std': std_t,
            'cost_trillion': np.mean(c_list),
            'avg_eta': np.mean(eta_list),
            'risk_score': total_risk,
            'env_risk_part': env_safety_risk,
            'temp_rise': env_metrics['temp_rise'],
            'co2e_million_tons': env_metrics['co2e_total'] / 1e6
        }

    def pareto_analysis(self, total_mass, n_points=30, alpha=0.0):
        ratios = np.linspace(0.0, 1.0, n_points)
        return [self.evaluate_strategy(total_mass, r, alpha=alpha) for r in ratios]