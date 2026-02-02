# engines.py
import numpy as np
from scipy.integrate import odeint
from typing import Dict, Tuple
from constants import *

C_HEAT = 12.0      # 混合层热容
ALPHA_BC = 1.8     # 辐射强迫系数
BETA_BC = 0.4      # 对数饱和参数
LAMBDA_F = 1.1     # 气候反馈参数
PHI_DEP = 0.15     # 表面沉降率
PSI0 = 0.02        # 基础洗脱率
KAPPA_T = 0.9      # 洗脱对温度的敏感度
A0_ALBEDO = 0.7    # 初始反照率

class EnvironmentalImpactModel:
    def __init__(self, params: EnvironmentalParameters):
        self.params = params

    def calculate_bc_pollution_index(self, bc_emission, rh=70.0):
        """
        bc_emission: 对应代码里的 x*20 (ug/m3)
        rh: 相对湿度
        """
        beta = 0.8
        gamma = 4.0
        f_rh = 1 + beta * (rh / 100.0)**gamma
        # 确保 self.params.AOD_KAPPA 存在
        aod = self.params.AOD_KAPPA * bc_emission * f_rh
        return aod
    
    def calculate_visibility(self, bc_emission):
        b_ray = 0.01 
        # 消光系数 b_ext = alpha * C^delta
        b_ext = self.params.VIS_ALPHA * (bc_emission**self.params.VIS_DELTA)
        visibility = 3.912 / (b_ext + b_ray)
        return visibility

    def calculate_stratospheric_disturbance(self, bc_x):
        # y = y0 + A * (1 - exp(-alpha * x))^gamma
        y0 = 0.1
        disturbance = y0 + self.params.STRAT_A * (1 - np.exp(-self.params.STRAT_ALPHA * bc_x))**self.params.STRAT_GAMMA
        return disturbance


    def _bc_climate_ode(self, y, t, bc_emission_rate):
            """
            ODE 定义
            y: 状态向量 [BC浓度, 辐射强迫, 温度变化]
            t: 时间点 (odeint 必须传递，即使函数内没用到)
            bc_emission_rate: 外部传入的参数
            """
            # 解构状态向量
            C_bc, RF, dT = y
            
            # 这里的物理常数应根据你的 constants.py 调整
            # 示例方程:
            dC_dt = bc_emission_rate - (1/self.params.BC_LIFETIME) * C_bc
            dRF_dt = self.params.RF_SENSITIVITY * dC_dt # 简化示例
            ddT_dt = (self.params.CLIMATE_SENSITIVITY * RF - dT) / self.params.OCEAN_HEAT_CAPACITY
            
            return [dC_dt, dRF_dt, ddT_dt]

    def calculate_rocket_emissions(self, payload_mass: float) -> Dict:
        if payload_mass <= 0:
            return {
                'co2_direct': 0, 'black_carbon': 0.0,
                'temperature_rise': 10, 'bc_climate_impact': 0.0,
                'total_co2e': 5, 'num_launches': 0
            }
        num_launches = np.ceil(payload_mass / self.params.ROCKET_PAYLOAD)
        total_fuel = num_launches * self.params.ROCKET_FUEL_CONSUMPTION # tons
        
        # 将总燃料排放转化为年均排放率 (E_val)，用于 ODE
        # 假设运输周期较长，这里简化为稳态排放输入
        bc_emission_rate = (total_fuel * self.params.ROCKET_BC_FACTOR) / 100.0 # 归一化输入

        # 求解 20 年后的气候稳态
        t_span = np.linspace(0, 20, 100)
        sol = odeint(self._bc_climate_ode, [0,0,0], t_span, args=(bc_emission_rate,))
        final_temp_rise = sol[-1, 1]

        return {
            'co2_direct': total_fuel * self.params.ROCKET_CO2_FACTOR,
            'black_carbon': total_fuel * self.params.ROCKET_BC_FACTOR,
            'temperature_rise': final_temp_rise,
            'bc_climate_impact': final_temp_rise * 100.0,
            'total_co2e': (total_fuel * self.params.ROCKET_CO2_FACTOR) + 
                          (total_fuel * self.params.ROCKET_BC_FACTOR * 900),
            'num_launches': num_launches
        }

    def calculate_hybrid_emissions(self, total_mass, rocket_ratio):
        m_rocket = total_mass * rocket_ratio
        m_ele = total_mass * (1 - rocket_ratio)
        
        res_r = self.calculate_rocket_emissions(m_rocket)
        co2_ele = m_ele * self.params.ELEVATOR_CO2_PER_TON / 1000.0
        
        return {
            'co2e_total': res_r['total_co2e'] + co2_ele,
            'temp_rise': res_r['temperature_rise'],
            'bc_impact': res_r['bc_climate_impact']
        }
    # engines.py 增加计算 eta 的方法
    def calculate_dynamic_eta(self, bc_x):
        """
        bc_x: 归一化黑碳强度 (0-5)
        """
        # 映射到 ug/m3
        bc_ug_m3 = bc_x * 20 
        
        # 计算各分量指标
        y_vis = self.calculate_visibility(bc_ug_m3)
        y_strat = self.calculate_stratospheric_disturbance(bc_x)
        
        # A. 传感器/控制损耗
        eta_vis_factor = (1 - np.exp(-y_vis / self.params.V_THRESHOLD))
        
        # B. 机械振动损耗 (基于平流层扰动)
        phi_bc = self.params.PHI_AMPLIFIER * y_strat
        dynamic_ratio = np.clip((phi_bc / self.params.T_MAX_MARGIN)**2, 0, 0.99)
        eta_dyn_factor = np.sqrt(1 - dynamic_ratio)
        
        # 返回耦合后的效率
        return self.params.ETA_BASE * eta_vis_factor * eta_dyn_factor