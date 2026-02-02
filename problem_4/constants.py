# constants.py
from dataclasses import dataclass

# --- 气候 ODE 系统常数 ---
C_HEAT = 12.0      # 混合层热容
ALPHA_BC = 1.8     # 辐射强迫系数
BETA_BC = 0.4      # 对数饱和参数
LAMBDA_F = 1.1     # 气候反馈参数
PHI_DEP = 0.15     # 表面沉降率
PSI0 = 0.02        # 基础洗脱率
KAPPA_T = 0.9      # 洗脱对温度的敏感度
A0_ALBEDO = 0.7    # 初始反照率
# constants.py 补充


@dataclass
class EnvironmentalParameters:
    # 火箭参数 (基于 Falcon Heavy)
    ROCKET_FUEL_CONSUMPTION = 4500.0  # tons fuel / launch
    ROCKET_PAYLOAD = 125.0            # tons payload / launch
    ANNUAL_ROCKET_LAUNCH =1188
    ROCKET_CO2_FACTOR = 3.0           # kg CO2 / kg fuel
    ROCKET_BC_FACTOR = 0.035          # kg BC / kg fuel
    
    # 太空电梯参数
    ELEVATOR_CAPACITY_YEAR = 179000.0 * 3  # 53.7k tons/year
    ELEVATOR_CO2_PER_TON = 100.0           # kg CO2e / ton (生命周期)
    
    # 经济与可靠性
    ROCKET_COST_PER_TON = 1.5e6      # 150万美元/吨
    ELEVATOR_COST_PER_TON = 2.0e4    # 2万美元/吨
    ROCKET_FAILURE_RATE = 0.02
    ELEVATOR_FAILURE_RATE = 0.10

    P_ROCKET_SUCCESS = 0.95 
    ETA = 0.85
    LAMBDA_SMALL_INIT = 20.0
    DECAY_RATE = 0.05
    DOWNTIME_SMALL = 0.02
    LAMBDA_BIG_FIXED = 0.3
    DOWNTIME_BIG = 0.5
    
    ROCKET_LOAD = 125           # payload per launch
    BS = 0.2                    # 自举系数
# 耦合模型参数
    ETA_BASE = 0.85          # 理想效率
    V_THRESHOLD = 2.0        # 能见度安全阈值 (km)
    T_MAX_MARGIN = 100.0     # 缆绳张力容余
    PHI_AMPLIFIER = 30.0     # 扰动放大系数
    
    # 环境子模型常数
    AOD_KAPPA = 0.02
    VIS_ALPHA = 0.05
    VIS_DELTA = 1.1
    STRAT_A = 1.2
    STRAT_ALPHA = 0.9
    STRAT_GAMMA = 2.3

    BC_LIFETIME = 1.0 / 365.0
    RF_SENSITIVITY = 0.5
    CLIMATE_SENSITIVITY = 0.8
    OCEAN_HEAT_CAPACITY = 10.0
    SIMULATION_YEARS = 20.0