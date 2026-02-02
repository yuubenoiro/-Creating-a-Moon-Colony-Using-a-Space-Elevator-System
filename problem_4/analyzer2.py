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

    def _single_stochastic_run(self, total_mass, r_usage, alpha=0.0):
        """
        单次随机仿真直到任务完成
        """
        t, m_accum, total_cost = 0, 0, 0
        cap_se = self.params.ELEVATOR_CAPACITY_YEAR # 初始年运力
        
        yearly_rocket_mass = []

        while m_accum < total_mass and t < 1000:
            # --- A. 火箭部分：二项分布随机发射 ---
            n_active = int(self.params.ANNUAL_ROCKET_LAUNCH * r_usage)
            # 模拟发射成功次数
            success_launches = np.random.binomial(n=n_active, p=self.params.P_ROCKET_SUCCESS)
            actual_rocket_mass = success_launches * self.params.ROCKET_LOAD
            
            # --- B. 电梯部分：随机故障与可用性损失 ---
            # 小故障：随时间衰减（可靠性增长模型）
            curr_lambda_small = self.params.LAMBDA_SMALL_INIT * np.exp(-self.params.DECAY_RATE * t)
            num_small = np.random.poisson(curr_lambda_small)
            # 大故障：固定概率
            num_big = np.random.poisson(self.params.LAMBDA_BIG_FIXED)
            
            # 计算总停机时间
            total_downtime = (num_small * self.params.DOWNTIME_SMALL) + (num_big * self.params.DOWNTIME_BIG)
            availability = max(0.1, 1.0 - total_downtime) # 最低维持10%运行能力
            
            # 有效运力 = 额定运力 * 效率系数 * 可用性
            effective_cap = cap_se * self.params.ETA * availability
            m_ele_to_moon = effective_cap * (1 - alpha)
            m_ele_to_upgrade = effective_cap * alpha
            
            # --- C. 状态与财务更新 ---
            m_accum += (actual_rocket_mass + m_ele_to_moon)
            yearly_rocket_mass.append(actual_rocket_mass)
            
            # 运力自举逻辑
            cap_se += (m_ele_to_upgrade * self.params.BS)
            
            # 成本计算：火箭按计划发射次数计费 + 电梯运行费
            total_cost += (n_active * self.params.ROCKET_LOAD * self.params.ROCKET_COST_PER_TON)
            total_cost += (m_ele_to_moon * self.params.ELEVATOR_COST_PER_TON)
            
            t += 1
            
        return t, total_cost / 1e12, np.mean(yearly_rocket_mass) if yearly_rocket_mass else 0

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