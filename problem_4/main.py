# main.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from constants import EnvironmentalParameters
from engines import EnvironmentalImpactModel
from analyzers import DecisionAnalyzer

# 1. 初始化系统
params = EnvironmentalParameters()
engine = EnvironmentalImpactModel(params)
analyzer = DecisionAnalyzer(engine)

# 2. 执行仿真 (针对1亿吨目标)
target_mass = 100_000_000 
results = analyzer.pareto_analysis(target_mass)
df = pd.DataFrame(results)

# 3. 可视化：帕累托前沿分析
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['time'], df['temp_rise'], c=df['cost'], 
                      cmap='viridis', s=df['rocket_ratio']*200, alpha=0.7)
plt.colorbar(scatter, label='Total Cost (Trillion $)')

# 标注关键点
plt.annotate('Eco-Strategy', xy=(df.iloc[0]['time'], df.iloc[0]['temp_rise']))
plt.annotate('Speed-Strategy', xy=(df.iloc[-1]['time'], df.iloc[-1]['temp_rise']))

plt.xlabel('Completion Time (Years)')
plt.ylabel('Global Temperature Rise Contribution (K)')
plt.title('Space Elevator vs Rocket: Strategic Trade-off Analysis')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("Pareto_Climate_Analysis.png", dpi=300)
plt.show()

print("仿真完成。数据已保存至 Pareto_Climate_Analysis.png")