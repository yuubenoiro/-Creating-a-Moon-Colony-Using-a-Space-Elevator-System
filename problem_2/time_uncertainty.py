import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model3 import stochastic_sim_v4 # 确保 model2.py 在路径中

def save_time_distribution(iterations=1000):
    # 运行模拟
    results = [stochastic_sim_v4([0.0, 0.4]) for _ in range(iterations)]
    t_list = [r[0] for r in results]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(t_list, kde=True, color='salmon', bins=25)
    plt.axvline(np.mean(t_list), color='black', linestyle='--', label=f'Mean: {np.mean(t_list):.1f}y')
    
    plt.title("A. Completion Time Uncertainty (Alpha=0)")
    plt.xlabel("Years")
    plt.ylabel("Frequency (Count)")
    plt.legend()
    plt.savefig("Time_Distribution_Indiv.png", dpi=300)
    plt.show()

save_time_distribution()