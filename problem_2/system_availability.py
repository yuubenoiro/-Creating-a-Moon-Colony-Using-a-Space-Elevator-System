import numpy as np
import matplotlib.pyplot as plt

# 提取故障逻辑
def get_availability_sample(years=100):
    avail_history = []
    for t in range(years):
        curr_lambda = 20.0 * np.exp(-0.05 * t)
        num_small = np.random.poisson(curr_lambda)
        num_big = np.random.poisson(0.3)
        # 对应 model2.py 中的可用性公式
        availability = max(0.1, 1.0 - (num_small * 0.02) - (num_big * 0.5))
        avail_history.append(availability)
    return avail_history

history = get_availability_sample()
plt.figure(figsize=(10, 5))
plt.plot(history, color='#2c3e50', linewidth=1.5)
plt.fill_between(range(100), history, 1.0, color='#2c3e50', alpha=0.1)

plt.title("B. System Availability Fluctuations (First 100 Years)")
plt.xlabel("Year")
plt.ylabel("Availability %")
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("Availability_Fluctuations_Indiv.png", dpi=300)
plt.show()