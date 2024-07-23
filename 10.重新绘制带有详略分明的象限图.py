import matplotlib.pyplot as plt
import random
import pysal
from splot.esda import lisa_cluster
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pysal.lib import weights
from esda.moran import Moran_Local

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

# 绘制四象限坐标图
# fig, ax = plt.subplots(figsize=(15, 15))
# gdf = pd.read_csv('city_lisa_categories_2带宽_0.1显著度.csv',encoding='GBK')
# 绘制散点图：X轴为观察值与全局平均值的差异，Y轴为空间滞后值与全局平均值的差异
# 这个用来绘制英文版的图
gdf = pd.read_csv('city_lisa_categories_2带宽_0.1显著度_英文版.csv',encoding='GBK')


# 计算全局平均值
global_mean = gdf['local_value'].mean()

# 计算横坐标（观察值与全局平均值的差异）
gdf['X'] = gdf['local_value'] - global_mean

# 计算纵坐标（空间滞后值与全局平均值的差异）
gdf['Y'] = gdf['neighbor_mean'] - global_mean


gdf = gdf[gdf['neighbor_mean'] != 0]
print(len(gdf))







# 示例数据
cities = gdf['city']
x_coords = gdf['X']
y_coords = gdf['Y']

# 需要突出显示的城市
highlight_cities = ['Sanya', 'Beijing', 'Shenzhen', 'Guangzhou', 'Nanjing', 'Wuhan', 'Chongqing', 'Weifang', 'Dalian', 'Suzhou']

# 按比例随机选择要标注的其他城市
other_cities = [city for city in cities if city not in highlight_cities]
random.seed(3)  # 固定随机种子以便结果可重复
selected_cities = random.sample(other_cities, int(0.3 * len(other_cities)))

# 创建图形
fig, ax = plt.subplots()

font = {'fontname':'Times New Roman'}


# 绘制所有城市的点
for i, city in enumerate(cities):
    if city in highlight_cities:
        # 突出显示的城市
        ax.scatter(x_coords[i], y_coords[i], color='red', marker='*', s=200, label=city)
        ax.text(x_coords[i] + 0.02, y_coords[i], city, fontsize=10, ha='left', fontname='Times New Roman')  # 添加偏移量
    else:
        # 其他城市
        ax.scatter(x_coords[i], y_coords[i], color='blue', marker='o', s=20)
        if city in selected_cities:
            if x_coords[i] < 0 and y_coords[i] < 0:
                if random.random() < 0.5:
                    ax.text(x_coords[i] + 0.01, y_coords[i], city, fontsize=8, ha='left', color='black', fontname='Times New Roman')  # 添加偏移量
            else:
                ax.text(x_coords[i] + 0.01, y_coords[i], city, fontsize=8, ha='left', color='black', fontname='Times New Roman')  # 添加偏移量


ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)

# 设置标题和标签
ax.set_title('')
ax.set_xlabel('Observed value deviation', fontname='Times New Roman', fontsize=12)
ax.set_ylabel('Neighborhood average deviation', fontname='Times New Roman', fontsize=12)
plt.savefig('四象限坐标图0721.png',dpi=600)
plt.show()











#
#
# count = 0
#
# for i in range(1, 10001):
#     if '8' in str(i):
#         count += 1
#
# print(f"在1到10000之间,包含数字8的数有{count}个。")