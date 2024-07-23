'''重新用距离阈值计算聚类模式LISA和四象限横纵轴'''
import geopandas as gpd
import pysal
from splot.esda import lisa_cluster
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pysal.lib import weights
from esda.moran import Moran_Local
import numpy as np

df = pd.read_csv('经度分组及比例_含样本总数.csv',encoding='GBK',usecols=['ratio','当地经度','当地纬度','出发市'])
# Display the first few rows of the data
print(df.head(),len(df))
# 假设你有一个DataFrame 'df'，其中包含了城市的人口落户率以及对应的空间位置（经纬度）


# 创建GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['当地经度'], df['当地纬度']))

# 创建基于距离的空间权重矩阵
distance_threshold = 2  # 根据需要调整距离阈值
w = pysal.lib.weights.DistanceBand.from_dataframe(gdf, threshold=distance_threshold, binary=False)

# 标准化权重
w.transform = 'r'

# 计算局部Moran指数
local_moran = Moran_Local(gdf['ratio'], w)

# 可视化结果
fig, ax = plt.subplots(figsize=(10, 10))
lisa_cluster(local_moran, gdf, p=0.1, ax=ax)
plt.show()

# 根据显著性筛选结果
significant = local_moran.p_sim < 0.1
gdf['lisa_category'] = 'ns'
lisa_categories = ['HH', 'LH', 'LL', 'HL']

gdf.loc[significant, 'lisa_category'] = pd.Categorical.from_codes(local_moran.q[significant] - 1, lisa_categories)

# 计算空间滞后值
gdf['neighbor_mean'] = pysal.lib.weights.spatial_lag.lag_spatial(w, gdf['ratio'])

# # 添加加权差异和
# weighted_diff_sum = np.zeros(len(gdf))
# for i in w.neighbors.keys():
#     local_value_i = gdf['ratio'].iloc[i]
#     for j in w.neighbors[i]:
#         local_value_j = gdf['ratio'].iloc[j]
#         diff = local_value_i - local_value_j
#         weight = w[i][j]
#         weighted_diff_sum[i] += diff * weight
#
# gdf['weighted_diff_sum'] = weighted_diff_sum



# 提取经纬度和城市名称
gdf['longitude'] = gdf.geometry.x
gdf['latitude'] = gdf.geometry.y
gdf['city'] = gdf['出发市']

# 添加局部Moran指数和邻居平均值到GeoDataFrame
gdf['local_value'] = gdf['ratio']




# 导出CSV
export_gdf = gdf[significant][['city', 'longitude', 'latitude', 'lisa_category', 'local_value', 'neighbor_mean']]
export_gdf.to_csv('city_lisa_categories_2带宽_0.1显著度.csv', index=False)









# 绘制四象限图0423

plt.rcParams["font.sans-serif"] = ['Microsoft YaHei']

# 绘制四象限坐标图
fig, ax = plt.subplots(figsize=(15, 15))
gdf = pd.read_csv('city_lisa_categories_2带宽_0.1显著度.csv',encoding='GBK')
# 绘制散点图：X轴为观察值与全局平均值的差异，Y轴为空间滞后值与全局平均值的差异


# 计算全局平均值
global_mean = gdf['local_value'].mean()

# 计算横坐标（观察值与全局平均值的差异）
gdf['X'] = gdf['local_value'] - global_mean

# 计算纵坐标（空间滞后值与全局平均值的差异）
gdf['Y'] = gdf['neighbor_mean'] - global_mean


gdf = gdf[gdf['neighbor_mean'] != 0]
print(len(gdf))

sc = ax.scatter(gdf['X'], gdf['Y'])

# 标注城市名称个别的，点要大一些，其他的就小一些也不标注
for x, y, label in zip(gdf['X'], gdf['Y'], gdf['city']):
    ax.text(x, y, label, fontsize=12)

# 绘制X=0和Y=0的轴线，分割四个象限
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)

# 设置坐标轴标签和图标题
ax.set_xlabel('观察值与全局平均值的差异')
ax.set_ylabel('空间滞后值与全局平均值的差异')
ax.set_title('四象限坐标图展示空间点的自相关性')
plt.show()
plt.savefig('四象限坐标图2带宽_0.05显著度.png',dpi=350)






# merge一下得到颜色的显著数据和全部城市数据，然后将其余的值设置为不显著，便于绘图


base_model = pd.read_csv('纯颜色块地图.csv',encoding='GBK')
print(base_model.head())


population = pd.read_csv("city_lisa_categories_2带宽_0.1显著度.csv",encoding='GBK')
print(population.head())

base_model = base_model.merge(population[['city', 'lisa_category']],
                                          on='city',
                                          how='left')


print(base_model.head(100))
base_model.to_csv('LISA色散图_0423.csv',index=False)














import geopandas as gpd
import matplotlib.pyplot as plt

# 计算全局平均值
global_mean = gdf['local_value'].mean()

# 计算横坐标（观察值与全局平均值的差异）
gdf['obs_diff'] = gdf['local_value'] - global_mean

# 计算纵坐标（空间滞后值与全局平均值的差异）
gdf['lag_diff'] = gdf['neighbor_mean'] - global_mean

# 创建四象限图
plt.figure(figsize=(8, 8))
plt.scatter(gdf['obs_diff'], gdf['lag_diff'], c=gdf['lisa_category'].cat.codes, cmap='viridis', alpha=0.7)
plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
plt.axvline(0, color='black', linestyle='--', linewidth=0.7)
plt.xlabel('Observation - Global Mean')
plt.ylabel('Spatial Lag - Global Mean')
plt.title('Quadrant Plot')
plt.show()
