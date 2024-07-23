
'''开始绘图，先画城市点'''
import matplotlib
matplotlib.use('TkAgg')  # 设置为Agg后端

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np


# Read the CSV file
data = pd.read_csv('经度分组及比例_含样本总数.csv',encoding='GBK')

# Display the first few rows of the data
print(data.head(),len(data))


line_colors = []
line_widths = []


# Load China map data
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
china = world[world.name == "China"]

# Define circle colors and sizes based on flow_rate
colors = []
sizes = []

for rate in data['ratio']:
    if rate <= 0.2:
        colors.append('blue')
        sizes.append(10)
    elif rate <= 0.45:
        colors.append('green')
        sizes.append(50)
    elif rate <= 0.7:
        colors.append('orange')
        sizes.append(150)
    else:
        colors.append('red')
        sizes.append(300)

# Plotting
fig, ax = plt.subplots(figsize=(15, 15))
china.plot(ax=ax, color='whitesmoke', edgecolor='black')

# Scatter plot for cities with flow_rate
ax.scatter(data['当地经度'], data['当地纬度'], c=colors, s=sizes, alpha=0.8, marker='o')



circle_labels = ["0.00% - 0.20%", "0.20% - 0.45%", "0.45% - 0.70%", "0.70% - 0.90%"]
circle_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=5),
                  plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=7),
                  plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10),
                  plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15),
                  ]

line_labels = ["Rank 1-10", "Rank 11-50", "Rank 51-300"]
line_handles = [plt.Line2D([0], [0], color='red', linewidth=2.5),
                plt.Line2D([0], [0], color='yellow', linewidth=2),
                plt.Line2D([0], [0], color='blue', linewidth=1.5)]

ax.legend(circle_handles, circle_labels, loc="upper left")


# Setting title and axis labels
# ax.set_title("Intercity mobility patterns 2022", fontsize=20)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.axis('off')
# Create a custom legend
plt.show()




'''统计并绘制六个层级城市的迁移平均意愿'''

city0 = ['上海市', '北京市', '广州市', '深圳市']

city1 = ['成都市', '重庆市', '杭州市', '武汉市', '西安市', '郑州市', '青岛市', '长沙市', '天津市', '苏州市', '南京市', '东莞市', '沈阳市', '合肥市', '佛山市']

city2 = ['昆明市', '福州市', '无锡市', '厦门市', '哈尔滨市', '长春市', '南昌市', '济南市', '宁波市', '大连市', '贵阳市', '温州市', '石家庄市', '泉州市', '南宁市',
         '金华市', '常州市', '珠海市',
         '惠州市', '嘉兴市', '南通市', '中山市', '保定市', '兰州市', '台州市', '徐州市', '太原市', '绍兴市', '烟台市', '廊坊市']

city3 = ['海口市', '汕头市', '潍坊市', '扬州市', '洛阳市', '乌鲁木齐市', '临沂市', '唐山市', '镇江市', '盐城市', '湖州市', '赣州市',
         '漳州市', '揭阳市', '江门市', '桂林市', '邯郸市', '泰州市', '济宁市', '呼和浩特市', '咸阳市', '芜湖市', '三亚市', '阜阳市',
         '淮安市', '遵义市', '银川市', '衡阳市', '上饶市', '柳州市', '淄博市', '莆田市', '绵阳市', '湛江市', '商丘市', '宜昌市', '沧州市', '连云港市',
         '南阳市', '蚌埠市', '驻马店市', '滁州市', '邢台市', '潮州市', '秦皇岛市', '肇庆市', '荆州市', '周口市', '马鞍山市', '清远市',
         '宿州市', '威海市', '九江市', '新乡市', '信阳市', '襄阳市', '岳阳市', '安庆市', '菏泽市', '宜春市', '黄冈市', '泰安市', '宿迁市',
         '株洲市', '宁德市', '鞍山市', '南充市', '六安市', '大庆市', '舟山市']
city4 = ['常德市', '渭南市', '孝感市', '丽水市', '运城市', '德州市', '张家口市', '鄂尔多斯市', '阳江市',
         '泸州市', '丹东市', '曲靖市', '乐山市', '许昌市', '湘潭市', '晋中市', '安阳市', '齐齐哈尔市',
         '北海市', '宝鸡市', '抚州市', '景德镇市', '延安市', '三明市', '抚顺市', '亳州市', '日照市', '西宁市',
         '衢州市', '拉萨市', '淮北市', '焦作市', '平顶山市', '滨州市', '吉安市', '濮阳市', '眉山市', '池州市',
         '荆门市', '铜仁市', '长治市', '衡水市', '铜陵市', '承德市', '达州市', '邵阳市', '德阳市', '龙岩市', '南平市',
         '淮南市', '黄石市', '营口市', '东营市', '吉林市', '韶关市', '枣庄市', '包头市', '怀化市', '宣城市', '临汾市',
         '聊城市', '梅州市', '盘锦市', '锦州市', '榆林市', '玉林市', '十堰市', '汕尾市', '咸宁市', '宜宾市', '永州市',
         '益阳市', '黔南布依族苗族自治州', '黔东南苗族侗族自治州', '恩施土家族苗族自治州', '红河哈尼族彝族自治州', '大理白族自治州', '大同市', '鄂州市', '忻州市', '吕梁市',
         '黄山市', '开封市', '郴州市', '茂名市', '漯河市', '葫芦岛市', '河源市', '娄底市', '延边朝鲜族自治州']


df_final = pd.read_csv('经度分组及比例_含样本总数.csv',encoding='GBK')

# Display the first few rows of the data
print(df_final.head(),len(df_final))


# 筛选出属于city0和city1的数据
df_city0 = df_final[df_final['出发市'].isin(city0)]
df_city1 = df_final[df_final['出发市'].isin(city1)]
df_city2 = df_final[df_final['出发市'].isin(city2)]
df_city3 = df_final[df_final['出发市'].isin(city3)]
df_city4 = df_final[df_final['出发市'].isin(city4)]
# 选择出发市不在city0和city1中的行
df_not_in_city0_or_city1 = df_final[~df_final['出发市'].isin(city0) & ~df_final['出发市'].isin(city1) & ~df_final['出发市'].isin(city2) & ~df_final['出发市'].isin(city3) & ~df_final['出发市'].isin(city4)]
print(len(df_not_in_city0_or_city1))

# 计算迁移意愿的平均值
avg_ratio_city0 = df_city0['ratio'].mean()
avg_ratio_city1 = df_city1['ratio'].mean()
avg_ratio_city2 = df_city2['ratio'].mean()
avg_ratio_city3 = df_city3['ratio'].mean()
avg_ratio_city4 = df_city4['ratio'].mean()
avg_ratio_city5 = df_not_in_city0_or_city1['ratio'].mean()

print(avg_ratio_city0,avg_ratio_city1,avg_ratio_city2,avg_ratio_city3,avg_ratio_city4,avg_ratio_city5)




# 数据


cities = ['超一线城市', '新一线城市', '二线城市', '三线城市', '四线城市', '五线城市']
values = [0.76642, 0.537027, 0.481671, 0.36035772, 0.3442299, 0.372076]  # 这里是示例数据，请根据实际数据进行调整


plt.rcParams["font.sans-serif"] = ['Microsoft YaHei']
plt.figure(figsize=(8,6))  # 设置图形大小
plt.bar(cities, values, color='blue', width=0.3)  # 创建柱状图

plt.xlabel('城市类型')  # x轴标签
plt.ylabel('转户意愿百分比(%)')  # y轴标签
plt.title('中国不同层级城市内流动人口的户籍转移意愿')  # 图形标题

plt.ylim(0, 0.8)  # 设置y轴范围
plt.tight_layout()  # 调整布局
# plt.grid(axis='y')  # 添加y轴网格线

plt.show()  # 显示图形




































'''绘制迁移率的moran散点图和指数计算'''
'''开始绘图，先画城市点'''
import matplotlib
matplotlib.use('TkAgg')  # 设置为Agg后端
import geopandas as gpd
import matplotlib.pyplot as plt
from pysal.lib import weights
from pysal.explore import esda
import pandas as pd
from splot.esda import plot_moran
from esda.moran import Moran_Local
import matplotlib.patches as mpatches
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pysal.lib import weights
from splot.esda import moran_scatterplot
from esda.moran import Moran_Local
import numpy as np

# Read the CSV file
df = pd.read_csv('经度分组及比例_含样本总数.csv',encoding='GBK',usecols=['ratio','当地经度','当地纬度','出发市'])

# Display the first few rows of the data
print(df.head(),len(df))
# 假设你有一个DataFrame 'df'，其中包含了城市的人口落户率以及对应的空间位置（经纬度）


# 创建GeoDataFrame，指定经纬度
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['当地经度'], df['当地纬度']))

# 创建空间权重矩阵
# 这里使用距离权重，以经纬度计算邻接关系。实际使用时，可能需要调整参数，比如距离的阈值。
w = weights.DistanceBand.from_dataframe(gdf, threshold=1.2, binary=True, silence_warnings=True)

# 标准化空间权重矩阵
w.transform = 'R'

# 计算Moran's I系数
mi = esda.moran.Moran(gdf['ratio'], w)

# 输出Moran's I结果
print(f"Moran's I: {mi.I}, 显著性水平 P-value: {mi.p_sim}")

# 绘制Moran散点图
# 计算局部Moran's I
local_moran = Moran_Local(gdf['ratio'], w)

# 将局部Moran's I结果添加到GeoDataFrame中
gdf['moran_local_I'] = local_moran.Is
gdf['moran_local_q'] = local_moran.q  # q值表示观察值所属的象限


# 定义颜色映射
color_map = {
    1: 'red',       # high-high
    2: 'lightblue', # low-high
    3: 'blue',      # low-low
    4: 'pink'       # high-low
}

# 为不同的空间相关性模式创建自定义图例项
legend_labels = {
    "high-high": "red",
    "low-high": "lightblue",
    "low-low": "blue",
    "high-low": "pink"
}

# 将q值映射到颜色
gdf['color'] = gdf['moran_local_q'].map(color_map)
# 将颜色信息回传给原始DataFrame
df['空间模式颜色'] = gdf['color']
df.to_csv('moran散点图数据(现在增加了一列颜色供arcgis).csv',index=False)


# 使用matplotlib.patches.Patch创建图例项
legend_handles = [mpatches.Patch(color=color, label=label) for label, color in legend_labels.items()]

# Load China map data
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
china = world[world.name == "China"]

plt.rcParams["font.sans-serif"] = ['Microsoft YaHei']

# Plotting
fig, ax = plt.subplots(figsize=(15, 15))
china.plot(ax=ax, color='whitesmoke', edgecolor='black')
gdf.plot(ax=ax, color=gdf['color'], markersize=25)

# 添加自定义图例
ax.legend(handles=legend_handles, title="空间相关性模式", loc="lower right", fontsize=20, title_fontsize=20)

# 去除绘图周围的边框、刻度线和标签
ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.tight_layout()
plt.show()

plt.savefig('moran散点图.png',dpi=350)









# 绘制四象限图
# Read the CSV file
df = pd.read_csv('经度分组及比例_含样本总数.csv',encoding='GBK',usecols=['ratio','当地经度','当地纬度','出发市'])
# Display the first few rows of the data
print(df.head(),len(df))
# 假设你有一个DataFrame 'df'，其中包含了城市的人口落户率以及对应的空间位置（经纬度）

# 创建GeoDataFrame，指定经纬度
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['当地经度'], df['当地纬度']))



# 计算全局平均值
global_mean = df['ratio'].mean()

# 计算每个点的观察值与全局平均值的差异并添加到df中
df['obs_minus_mean'] = df['ratio'] - global_mean

# 使用libpysal计算空间滞后值
# 确保空间权重矩阵w已经根据gdf创建并进行了行标准化
# 创建空间权重矩阵和进行行标准化
w = weights.DistanceBand.from_dataframe(gdf, threshold=1.2, binary=True, silence_warnings=True)
w.transform = 'R'
# 计算空间滞后值
lag_value = weights.spatial_lag.lag_spatial(w, df['ratio'])

# 计算空间滞后值与全局平均值的差异（net_lag_value）
net_lag_value = lag_value - global_mean

# 将计算结果添加到gdf中
gdf['obs_minus_mean'] = df['obs_minus_mean']  # 观察值与全局平均值的差异
gdf['net_lag_value'] = net_lag_value  # 空间滞后值与全局平均值的差异
df_gdf = pd.DataFrame(gdf.drop(columns='geometry'))
df_gdf.to_csv('四象限坐标图数据.csv',index=False,encoding='GBK')





plt.rcParams["font.sans-serif"] = ['Microsoft YaHei']

# 绘制四象限坐标图
fig, ax = plt.subplots(figsize=(15, 15))
gdf = pd.read_csv('city_lisa_categories2.csv',encoding='GBK')
# 绘制散点图：X轴为观察值与全局平均值的差异，Y轴为空间滞后值与全局平均值的差异
sc = ax.scatter(gdf['X'], gdf['Y'])

# 标注城市名
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
plt.savefig('四象限坐标图0329.png',dpi=350)








# 随机选取一些城市进行绘图
import pandas as pd


# 读取Excel文件
df = pd.read_excel('随机一下选取.xlsx', engine='openpyxl')

# 从df中随机选取100行
sampled_df = df.sample(n=100)
#
# # 如果DataFrame行数少于100，并且你希望总共选择出100行（允许重复选择行）
# sampled_df = df.sample(n=100, replace=True)

# 如果你需要将选取的行保存回一个新的Excel文件
sampled_df.to_csv('随机选取之后的城市列表.csv', index=False)














# lisa图重新绘制03.29

import geopandas as gpd
import pysal
from splot.esda import lisa_cluster
from esda.moran import Moran_Local
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.pyplot as plt
from pysal.lib import weights
from pysal.explore import esda
import pandas as pd
from splot.esda import plot_moran
from esda.moran import Moran_Local
import matplotlib.patches as mpatches
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pysal.lib import weights
from splot.esda import moran_scatterplot
from esda.moran import Moran_Local
import numpy as np

df = pd.read_csv('经度分组及比例_含样本总数.csv',encoding='GBK',usecols=['ratio','当地经度','当地纬度','出发市'])
# Display the first few rows of the data
print(df.head(),len(df))
# 假设你有一个DataFrame 'df'，其中包含了城市的人口落户率以及对应的空间位置（经纬度）

# 创建GeoDataFrame，指定经纬度
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['当地经度'], df['当地纬度']))


# 设定用于分析的变量
variable = 'ratio'

# 创建空间权重矩阵
w = pysal.lib.weights.Queen.from_dataframe(gdf)

# 标准化权重
w.transform = 'r'

# 计算局部Moran指数
local_moran = Moran_Local(gdf[variable], w)

# 使用splot库中的lisa_cluster函数来可视化结果
fig, ax = plt.subplots(figsize=(10, 10))
lisa_cluster(local_moran, gdf, p=0.05, ax=ax)  # p参数设置显著性水平

plt.show()

significant = local_moran.p_sim < 0.05

# Initialize the 'lisa_category' column with 'ns'
gdf['lisa_category'] = 'ns'

# Define the LISA categories
lisa_categories = ['HH', 'LH', 'LL', 'HL']  # Moran's I quadrants start from 1, so we add 'ns' at the beginning

# Update the 'lisa_category' for significant values
gdf.loc[significant, 'lisa_category'] = pd.Categorical.from_codes(local_moran.q[significant] - 1, lisa_categories)

# 提取经纬度和城市名称
gdf['longitude'] = gdf.geometry.x
gdf['latitude'] = gdf.geometry.y
gdf['city'] = gdf['出发市']

# 添加局部Moran指数和邻居平均值到GeoDataFrame
gdf['local_value'] = gdf['ratio']
# 计算空间滞后值
gdf['neighbor_mean'] = weights.lag_spatial(w, gdf['ratio'])



# 初始化加权差异和的数组
weighted_diff_sum = np.zeros(len(gdf))

# 遍历每个空间单元
for i in w.neighbors.keys():
    # 获取空间单元i的值
    local_value_i = gdf['local_value'].iloc[i]

    # 计算空间单元i与其所有邻居的差异，并进行加权
    for j in w.neighbors[i]:
        local_value_j = gdf['local_value'].iloc[j]
        diff = local_value_i - local_value_j
        weight = w[i][j]
        weighted_diff_sum[i] += diff * weight

# 将加权差异和添加到GeoDataFrame中
gdf['weighted_diff_sum'] = weighted_diff_sum

# 选择需要的列
export_gdf = gdf[significant][['city', 'longitude', 'latitude', 'lisa_category', 'local_value', 'neighbor_mean','weighted_diff_sum']]

# 保存到CSV
export_gdf.to_csv('city_lisa_categories3.csv', index=False)








'''
一些冗余工作，比如说上面的表只包含显著的城市，不利于画图，那么我要对所有城市进行补全
'''
import pandas as pd
dfA = pd.read_csv('city_lisa_categories2.csv',encoding='GBK')
dfB = pd.read_csv('moran散点图数据(现在增加了一列颜色供arcgis).csv')

# 筛选出在dfA中不存在的城市
filtered_dfB = dfB[~dfB['出发市'].isin(dfA['city'])]

print(filtered_dfB)

filtered_dfB.to_csv('not_significant_table.csv',index=False)

















