# analyzer2.py
import numpy as np
from engines import EnvironmentalImpactModel

class DecisionAnalyzerV2:
    """
    随机决策分析器 (Stochastic Decision Analyzer)
    集成蒙特卡洛模拟，考虑随机故障、停机时间以及火箭发射成功率。
    """
    def __init__(self, env_model: EnvironmentalImpactModel):
        self.model = env_model
        self.params = env_model.params

# analyzer2.py (部分代码修改)

    def _single_stochastic_run(self, total_mass, r_usage, alpha=0.0):
        t, m_accum, total_cost = 0, 0, 0
        cap_se = self.params.ELEVATOR_CAPACITY_YEAR
        
        while m_accum < total_mass and t < 1000:
            # --- A. 火箭发射 (产生黑碳) ---
            n_active = int(self.params.ANNUAL_ROCKET_LAUNCH * r_usage)
            success_launches = np.random.binomial(n=n_active, p=self.params.P_ROCKET_SUCCESS)
            actual_rocket_mass = success_launches * self.params.ROCKET_LOAD
            
            # --- B. 环境反馈：计算动态 ETA ---
            # 定义当前黑碳强度 x 为火箭启用率的函数 (简化模型)
            bc_x = r_usage * 5.0  # 将 0-1 映射到 0-5 的强度范围
            dynamic_eta = self.model.calculate_dynamic_eta(bc_x)
            
            # --- C. 电梯运力 (受动态 ETA 影响) ---
            # 故障逻辑
            num_small = np.random.poisson(self.params.LAMBDA_SMALL_INIT * np.exp(-self.params.DECAY_RATE * t))
            num_big = np.random.poisson(self.params.LAMBDA_BIG_FIXED)
            availability = max(0.1, 1.0 - (num_small * 0.02 + num_big * 0.5))
            
            # 【核心修改点】：使用 dynamic_eta 替代静态 ETA
            effective_cap = cap_se * dynamic_eta * availability
            
            # --- D. 运输与更新 ---
            m_ele_to_moon = effective_cap * (1 - alpha)
            m_accum += (actual_rocket_mass + m_ele_to_moon)
            cap_se += (effective_cap * alpha * self.params.BS)
            t += 1
            
        return t, total_cost, np.mean(dynamic_eta) # 返回平均效率用于分析

    def evaluate_strategy(self, total_mass, r_usage, alpha=0.0, iterations=50):
        """
        通过蒙特卡洛模拟进行多次采样，获取统计指标
        """
        t_list, c_list, r_mass_list = [], [], []
        
        for _ in range(iterations):
            t, c, r_m = self._single_stochastic_run(total_mass, r_usage, alpha)
            t_list.append(t)
            c_list.append(c)
            r_mass_list.append(r_m)
            
        mean_t = np.mean(t_list)
        std_t = np.std(t_list)
        
        # 风险评分定义：[时间波动率(CV) * 权重] + [单点故障冗余风险]
        # 当 std_t/mean_t 越高，说明系统越不稳定
        reliability_risk = (std_t / mean_t) * 150 if mean_t > 0 else 0
        redundancy_risk = 40 * (1 - 4 * r_usage * (1 - r_usage))
        total_risk_score = np.clip(reliability_risk + redundancy_risk, 0, 100)
        
        # 环境影响：基于平均年发射份额计算
        avg_r_mass = np.mean(r_mass_list)
        # 粗略估算平均运输份额
        avg_annual_total = total_mass / mean_t
        actual_share = avg_r_mass / avg_annual_total if avg_annual_total > 0 else 0
        
        env = self.model.calculate_hybrid_emissions(total_mass, actual_share)

        return {
            'rocket_ratio': r_usage,
            'alpha': alpha,
            'time_years': mean_t,
            'time_std': std_t,
            'cost_trillion': np.mean(c_list),
            'risk_score': total_risk_score,
            'temp_rise': env['temp_rise'],
            'co2e_million_tons': env['co2e_total'] / 1e6
        }

    def pareto_analysis(self, total_mass, n_points=30, alpha=0.0):
        """
        帕累托分析接口
        """
        ratios = np.linspace(0.0, 1.0, n_points)
        return [self.evaluate_strategy(total_mass, r, alpha=alpha) for r in ratios]