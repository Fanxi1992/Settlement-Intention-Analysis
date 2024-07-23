import numpy as np
import pandas as pd

# 假设 param_estimates_1 和 param_estimates_2 分别代表两组参数估计的值
df = pd.read_csv('18张参数空间分布图的合并表 -计算变异系数0329.csv',encoding='GBK')  # 如果你的数据在CSV文件中

# 定义计算变异系数的函数
def coefficient_of_variation(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    return std_dev


# 遍历DataFrame的每一列
for column in df.columns:
    # 筛选出不是'not_significant'的值
    filtered_data = df[column][df[column] != 'Not significant']

    # 将筛选后的数据转换为数值类型
    numerical_data = pd.to_numeric(filtered_data)

    # 计算变异系数
    cv = coefficient_of_variation(numerical_data)
    print(f"变异系数 for {column}: {cv}")








'''
绘图
'''
import matplotlib
matplotlib.use('TkAgg')  # 设置为Agg后端
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_excel('局部估计系数分布图.xlsx')
# 假设df是你的DataFrame
# 示例数据
print(df)

# 创建画布和轴对象
fig, ax1 = plt.subplots(figsize=(15, 10))
# 绘制方差柱状图
ax1.bar(df['variable'], df['std'], color='b', label='Standard Deviation', width=0.5)
ax1.set_xlabel('Variable')
ax1.set_ylabel('Standard Deviation', color='b')
ax1.tick_params(axis='y', labelcolor='b', labelsize=20)
# plt.show()
# 创建第二个Y轴
ax2 = ax1.twinx()

# 绘制显著度折线图
ax2.plot(df['variable'], df['sig'], color='r', marker='o', label='Significance',lw=2)
ax2.set_ylabel('Significance', color='r')
ax2.tick_params(axis='y', labelcolor='r', labelsize=20)

# 添加图例
ax1.legend(loc='upper left',fontsize=20)
ax2.legend(loc='upper right',fontsize=20)

ax1.set_xticklabels(df['variable'], rotation=45,fontsize=15)
plt.tight_layout()
# 显示图表
plt.show()











df=pd.read_excel('局部估计系数分布图.xlsx')
# 假设df是你的DataFrame
# 示例数据
print(df)


plt.rcParams["font.sans-serif"] = ['Microsoft YaHei']
# 创建画布和轴对象
fig, ax1 = plt.subplots(figsize=(15, 10))
# 绘制方差柱状图
ax1.bar(df['variable'], df['sig'], color='black', label='Significant parameters number', width=0.5)  #,lw=2
ax1.set_xlabel('')
ax1.set_ylabel('number', color='black', fontsize=25)
ax1.tick_params(axis='y', labelcolor='black', labelsize=20)
# plt.show()
# 创建第二个Y轴
ax2 = ax1.twinx()

# 绘制显著度折线图

ax2.plot(df['variable'], df['std'], color='r', marker='s', markersize=10,label='Standard Deviation',lw=2) # , color='r', marker='o', label='Significance',lw=2)
# ax2.set_xlabel('Variable')
ax2.set_ylabel('Standard Deviation', color='black',fontsize=25)
ax2.tick_params(axis='y', labelcolor='black', labelsize=20)
# 添加图例
ax1.legend(loc='upper left',fontsize=20)
ax2.legend(loc='upper right',fontsize=20)

ax1.set_xticklabels(df['variable'], rotation=45,fontsize=15, ha='right')
plt.tight_layout()
plt.savefig('局部估计系数分布图0619.jpg', dpi=600)
# 显示图表
plt.show()


