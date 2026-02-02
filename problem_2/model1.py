import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ========================
# 1. 参数设置 (Alpha = 0)
# ========================
ALPHA = 0.0
R_USAGE = 0.4  # 固定火箭利用率
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
# 其他物理参数沿用 model2.py ...

def single_scenario_analysis(iters=2000):
    t_list = []
    c_list = []
    
    # 为了展示“运力损耗时序”，我们单独记录一次模拟的细节
    example_availability = []
    
    for i in range(iters):
        t, m_accum = 0, 0
        cap_se = INITIAL_CAP
        temp_avail = []
        
        while m_accum < TARGET_MASS and t < 1000:
            # 火箭二项分布
            n_active = int(N_ROCKET_WINDOW * R_USAGE)
            success = np.random.binomial(n=n_active, p=P_ROCKET_SUCCESS)
            
            # 电梯故障逻辑
            curr_lambda = LAMBDA_SMALL_INIT * np.exp(-DECAY_RATE * t)
            num_small = np.random.poisson(curr_lambda)
            num_big = np.random.poisson(LAMBDA_BIG_FIXED)
            availability = max(0.1, 1.0 - (num_small * DOWNTIME_SMALL) - (num_big * DOWNTIME_BIG))
            
            if i == 0: # 记录第一次模拟作为典型案例
                temp_avail.append(availability)
                
            m_accum += (success * ROCKET_LOAD + cap_se * ETA * availability)
            t += 1
            
        t_list.append(t)
        if i == 0: example_availability = temp_avail

    return np.array(t_list), example_availability

# 执行分析
t_dist, avail_history = single_scenario_analysis()

# ========================
# 2. 可视化
# ========================
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2)

# 图 A: 完工时间分布
ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(t_dist, kde=True, color='#e74c3c', ax=ax1)
ax1.axvline(t_dist.mean(), color='black', linestyle='--', label=f'Mean: {t_dist.mean():.1f}y')
ax1.set_title("A. Completion Time Uncertainty (Alpha=0)", fontsize=14, fontweight='bold')
ax1.set_xlabel("Years")
ax1.legend()

# 图 B: 典型运力可用性波动 (前100年)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(range(len(avail_history[:100])), avail_history[:100], color='#34495e', lw=1.5)
ax2.fill_between(range(len(avail_history[:100])), avail_history[:100], 1, color='#34495e', alpha=0.1)
ax2.set_title("B. System Availability Fluctuations (First 100 Years)", fontsize=14, fontweight='bold')
ax2.set_ylabel("Availability %")
ax2.set_xlabel("Year")
ax2.set_ylim(0, 1.1)

# 图 C: 累计概率曲线 (CDF)
ax3 = fig.add_subplot(gs[1, :])
sns.ecdfplot(t_dist, color='#e74c3c', ax=ax3, lw=2)
ax3.axhline(0.95, color='gray', linestyle=':', label='95% Confidence')
ax3.set_title("C. Cumulative Probability of Completion", fontsize=14, fontweight='bold')
ax3.set_xlabel("Years")
ax3.set_ylabel("Probability")
ax3.legend()

plt.tight_layout()
plt.savefig("Alpha0_Risk_Analysis.png", dpi=300)
plt.show()