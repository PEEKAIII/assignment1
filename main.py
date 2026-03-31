import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family']='SimHei'
plt.rcParams['axes.unicode_minus']=False
filepath = 'data/TopRight_20230803.txt'
# ---------------------- 1. 读取并清洗数据 ----------------------
# 跳过注释行（以#开头的行），读取数据
# 数据以制表符\t分隔，手动指定列名
col_names = [
    "Event", "Time", "Date", "TimeStamp_ms", "ADC1", "ADC2",
    "SiPM_mV", "Temp_C", "Pressure_Pa", "DeadTime_us", "Coincident", "ID"
]

# 读取数据：跳过以#开头的注释行
df = pd.read_csv(
    filepath,
    delimiter="\t",  # 数据是制表符分隔
    comment="#",     # 自动跳过#开头的行
    names=col_names,
    index_col=False,
    #on_bad_lines="skip"  # 跳过格式错误的行
)

# 提取核心数据：时间戳（毫秒）
timestamp = df["TimeStamp_ms"].values
print(df.head(5))
print(timestamp[0:5])
# ---------------------- 问题1：数组元素个数（信号事件数） ----------------------
n_events = len(timestamp)
print("="*60)
print(f"1. 信号事件总数（数组元素个数）: {n_events}")

# ---------------------- 问题2：TimeStamp直方图（100个bins）+ 分布形状 ----------------------
plt.figure(figsize=(10, 5))
n, bins, patches = plt.hist(timestamp, bins=100, edgecolor="black", alpha=0.7)
plt.title("TimeStamp [ms] 分布直方图 (100 bins)", fontsize=14)
plt.xlabel("TimeStamp (毫秒)")
plt.ylabel("事件数量")
plt.grid(alpha=0.3)
plt.show()

# 分布形状判断（均匀分布/泊松分布）
print(f"2. 直方图分布形状: 近似**均匀分布**（事件在时间上均匀发生）")

# ---------------------- 问题3：总采集时间 + 信号事件率（每秒） ----------------------
# 最后一个TimeStamp = 总采集时间（毫秒）
total_duration_ms = timestamp[-1]
total_duration_s = total_duration_ms / 1000  # 转换为秒
event_rate = n_events / total_duration_s    # 事件率（个/秒）

print(f"3. 总采集时间: {total_duration_s:.2f} 秒")
print(f"   信号事件率: {event_rate:.4f} 个/秒")

# ---------------------- 问题4：相对统计不确定度（变异系数） ----------------------
# 步骤1：100个bin的时间宽度（每个bin对应的秒数）
bin_width_ms = (timestamp.max() - timestamp.min()) / 100
bin_width_s = bin_width_ms / 1000

# 步骤2：计算每个bin的事件率（个/秒）
bin_rates = n / bin_width_s

# 步骤3：计算100个率的均值和标准差
mean_rate = np.mean(bin_rates)
std_rate = np.std(bin_rates, ddof=1)  # 样本标准差

# 步骤4：相对不确定度 = 标准差 / 均值（变异系数）
relative_uncertainty = std_rate / np.abs(mean_rate)

print(f"4. 100个bin的平均事件率: {mean_rate:.4f} 个/秒")
print(f"   标准差: {std_rate:.4f}")
print(f"   相对统计不确定度（变异系数）: {relative_uncertainty:.6f}")
print("="*60)