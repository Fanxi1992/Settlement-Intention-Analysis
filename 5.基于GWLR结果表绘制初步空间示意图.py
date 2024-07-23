# 首先，导入必要的库并读取用户提供的CSV文件。
import matplotlib

matplotlib.use('TkAgg')  # 设置为Agg后端
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from pysal.lib import weights
import statsmodels.api as sm
import pysal as ps
import esda
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 读取CSV文件
file_path = '全样本GWLR_regression_results_加上城市层面变量之后0315_去掉一个奇异值.csv'
df = pd.read_csv(file_path,encoding='GBK')

# 显示数据的前几行，以便了解数据结构和列名称
print(df.head())
print(len(df))

# 看看有多少显著的，对于每一个变量，计算P值小于0.1的行数
select_column = ['P值_截距项','P值_性别_女', 'P值_年龄段_中年', 'P值_年龄段_青年','P值_受教育程度_中学学历','P值_受教育程度_大学及以上学历','P值_户口性质_农业','P值_婚姻状况_已婚',
          'P值_感觉被看不起_同意','P值_跨越尺度_省内跨市','P值_跨越尺度_跨省','P值_distance','P值_所在城市常住人口（万人）','P值_所在城市GDP_亿元',
          'P值_所在城市在岗职工平均工资_元','P值_所在城市第三产业比重','P值_所在城市PM2.5','P值_流动时长','P值_同住的家庭成员人数         ']
df_significant = df[select_column]
# 计算显著性的行数，判断该变量是否有空间研究价值
count_less_than_01 = (df_significant < 0.1).sum()
print(count_less_than_01)


# 计算每一个显著变量的描述性统计
# 准备一个字典来存储结果
stats = {}
for col in select_column:
    # 生成对应的实际参数列名
    param_col = col.replace('P值_', '参数_')  # 假设实际参数值的列名是去掉'P值_'后的名称
    # 选择显著性检验通过的行
    significant_rows = df[col] < 0.1
    if significant_rows.any():  # 如果存在显著性行
        # 计算描述性统计
        stats[param_col] = {
            'min': df.loc[significant_rows, param_col].min(),
            'max': df.loc[significant_rows, param_col].max(),
            'mean': df.loc[significant_rows, param_col].mean()
        }
# 打印每个参数的统计结果
for param, stat in stats.items():
    print(f"{param}: Min={stat['min']}, Max={stat['max']}, Mean={stat['mean']}")


# 查看和剔除异常值
col = 'P值_所在城市GDP_亿元'
param_col = '参数_所在城市GDP_亿元'
param_col2 = ['城市位置','参数_所在城市GDP_亿元']
# 先根据param_col列的值进行降序排序
df_sorted = df.sort_values(by=param_col, ascending=False)
significant_rows = df_sorted[col] < 0.1
if significant_rows.any():  # 如果存在显著性行
    print(df_sorted.loc[significant_rows,param_col2])



# 假设你想删除的元素数量
num_to_delete = 3
# 获取显著性行的索引
significant_indices = df_sorted[significant_rows].index
# 如果有需要删除的元素
if len(significant_indices) > num_to_delete:
    # 删除最小值所在的行，这里使用了尾部切片来保留除了最后n个元素之外的所有元素
    # 因为df_sorted已经是按param_col降序排列的，所以最小的元素会在最后
    significant_indices_to_keep = significant_indices[:-num_to_delete]
    # 使用更新后的索引来选取数据
    data_to_print = df_sorted.loc[significant_indices_to_keep, param_col]
else:
    # 如果没有足够的元素可以删除，就打印所有符合条件的元素
    data_to_print = df_sorted.loc[significant_indices, param_col]
print(data_to_print)
print(param_col)
stats[param_col] = {
            'min': data_to_print.min(),
            'max': data_to_print.max(),
            'mean': data_to_print.mean()
        }
# 打印每个参数的统计结果
for param, stat in stats.items():
    print(f"{param}: Min={stat['min']}, Max={stat['max']}, Mean={stat['mean']}")







'''开始绘制指定变量的空间异质性'''
gwr_model = pd.read_csv('全样本GWLR_regression_results_加上城市层面变量之后.csv')
print(gwr_model.head(100))

# 去除左右方括号
gwr_model['城市位置'] = gwr_model['城市位置'].str.replace('[', '').str.replace(']', '')

''' 从 GWR 模型中提取参数估计和 p 值:'''
# Assuming 'distance' is the variable of interest and it's the first column in X
variable_index = '参数_户口性质_农业'
significant_index = 'P值_户口性质_农业'
gwr_model[['Longitude_index', 'latitude_index']] = gwr_model['城市位置'].str.split(r'\s+', expand=True).iloc[:, :2].astype(float)
Longitude_index = 'Longitude_index'
latitude_index = 'latitude_index'
print(gwr_model['Longitude_index'].head(300))
print(gwr_model['latitude_index'])

# Extract parameter estimates and p-values for 'distance' from the GWR model
param_estimates = gwr_model[variable_index]
Longitude = gwr_model[Longitude_index]
Latitude = gwr_model[latitude_index]
pvalues_distance = gwr_model[significant_index]
print(pvalues_distance)

# Find indices where p-value is less than 0.10
# 找到p值小于0.1的索引
significant_mask = pvalues_distance < 0.1
# 提取这些索引对应的经度、纬度和参数估计
jingdu_coords = Longitude[significant_mask]
print(jingdu_coords)

weidu_coords = Latitude[significant_mask]
significant_params = param_estimates[significant_mask]


significant_data = pd.DataFrame({
    'Longitude': jingdu_coords,
    'Latitude': weidu_coords,
    'Parameter Value': significant_params,
})

print(significant_data)
significant_data.to_csv('significant_data_参数_户口性质_农业.csv', index=False)

desc_stats = significant_data.describe()
desc_stats.to_csv('significant_data_{}_from_{}_参数描述性统计结果.csv'.format(variable_index, sigh_for_significant_data),
                  index=False)
'''
    Longitude   Latitude  Parameter Value Start City
0  117.362499  28.749783        13.563012         上饶
1  118.310951  35.292000        16.942450         临沂
2  103.580582  29.174616        20.150308         乐山
3  105.392094  26.123164        25.343645        六盘水
4  104.856604  29.614104        19.133537         内江

'''

'''创建自定义的红色色条'''

# 创建一个只有红色渐变的色彩映射
colors = [(1, 0, 0), (1, 0.95, 0.95)]  # R -> R
n_bins = 100  # Discretizes the interpolation into bins
cmap_name = 'custom1'
red_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)



'''geopandas 来绘制地图，并根据参数值的大小为每个点分配颜色。'''
# 创建一个 GeoDataFrame，其中 'geometry' 列包含每个点的坐标
gdf = gpd.GeoDataFrame(significant_data,
                       geometry=gpd.points_from_xy(significant_data.Longitude, significant_data.Latitude))

# 中国边界地图
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
china = world[world.name == "China"]

# 绘制地图
fig, ax = plt.subplots(figsize=(15, 15))
china.plot(ax=ax, color='white', edgecolor='black')
gdf.plot(column='Parameter Value', ax=ax, legend=True, markersize=50, cmap="Reds")
plt.title("Spatial Heterogeneity for nongyerenkou")
plt.show()













'''合并城市名给老曹画arcgis'''
file_path = 'logit模型数据表（不包含性别交叉项)+城市层面+去空值并标准化.csv'
df = pd.read_csv(file_path)
print(df.head())
file_path = 'significant_data_参数_户口性质_农业.csv'
df1 = pd.read_csv(file_path, encoding='GBK')
print(df1.head())
print(len(df1))
df_unique = df.drop_duplicates(subset=['当地经度'])
print(len(df_unique))


df1 = df1.merge(df_unique[['当地经度', '出发市']],
                                          on='当地经度',
                                          how='left')
df1.rename(columns={'出发市': '城市'}, inplace=True)
print(len(df1))


df1.to_csv('significant_data_参数_户口性质_农业 _带城市名.csv', index=False)
























'''重新使用在推拉模型中完成的代码进一步提高significant表格和绘图的自动化速度'''

# 读取CSV文件
# 读取CSV文件
file_path_list = ['全样本GWLR_regression_results_加上城市层面变量之后.csv'
                  ]

for file_path in file_path_list:
    df = pd.read_csv(file_path)
    # print(df.head())
    sigh_for_significant_data = file_path.replace('/', '_')
    print(sigh_for_significant_data)

    select_column = ['P值_性别_女', 'P值_年龄段_中年', 'P值_年龄段_壮年', 'P值_年龄段_老年', 'P值_受教育程度_中学学历',
                           'P值_受教育程度_大学专科', 'P值_受教育程度_本科及以上学历', 'P值_户口性质_农业', 'P值_婚姻状况_已婚',
                           'P值_婚姻状况_离婚', 'P值_本次流动原因_家庭类', 'P值_本次流动原因_工作类', 'P值_感觉被看不起_同意',
                     'P值_distance', 'P值_所在城市常住人口（万人）', 'P值_所在城市GDP（亿元）', 'P值_所在城市一般公共预算支出（万元）',
                     'P值_所在城市教育支出（万元）', 'P值_所在城市科技支出（万元）', 'P值_所在城市二产比重'
                     ]
    df_significant = df[select_column]
    # 计算显著性的行数，判断该变量是否有空间研究价值
    count_less_than_01 = (df_significant < 0.1).sum()
    print(count_less_than_01)

    '''开始绘制指定变量的空间异质性'''
    gwr_model = df

    # 去除左右方括号
    gwr_model['城市位置'] = gwr_model['城市位置'].str.replace('[', '').str.replace(']', '')

    ''' 从 GWR 模型中提取参数估计和 p 值:'''
    # Assuming 'distance' is the variable of interest and it's the first column in X
    gwr_model[['Longitude_index', 'latitude_index']] = gwr_model['城市位置'].str.split(r'\s+', expand=True).iloc[:,
                                                       :2].astype(float)
    # print(gwr_model['Longitude_index'].head(300))
    # print(gwr_model['latitude_index'])
    #
    print(gwr_model.head(100))

    ''' 从 GWR 模型中提取参数估计和 p 值:'''
    # Assuming 'distance' is the variable of interest and it's the first column in X
    Longitude_index = 'Longitude_index'
    latitude_index = 'latitude_index'
    variable_index_list = ['参数_性别_女', '参数_年龄段_中年', '参数_年龄段_壮年', '参数_年龄段_老年', '参数_受教育程度_中学学历',
                           '参数_受教育程度_大学专科', '参数_受教育程度_本科及以上学历', '参数_户口性质_农业', '参数_婚姻状况_已婚',
                           '参数_婚姻状况_离婚', '参数_本次流动原因_家庭类', '参数_本次流动原因_工作类', '参数_感觉被看不起_同意']
    significant_index_list = ['P值_性别_女', 'P值_年龄段_中年', 'P值_年龄段_壮年', 'P值_年龄段_老年', 'P值_受教育程度_中学学历',
                           'P值_受教育程度_大学专科', 'P值_受教育程度_本科及以上学历', 'P值_户口性质_农业', 'P值_婚姻状况_已婚',
                           'P值_婚姻状况_离婚', 'P值_本次流动原因_家庭类', 'P值_本次流动原因_工作类', 'P值_感觉被看不起_同意']
    resist_index_list = ['参数_distance', '参数_所在城市常住人口（万人）', '参数_所在城市GDP（亿元）', '参数_所在城市一般公共预算支出（万元）',
                         '参数_所在城市教育支出（万元）','参数_所在城市科技支出（万元）','参数_所在城市二产比重']
    significant_resist_list = ['P值_distance', 'P值_所在城市常住人口（万人）', 'P值_所在城市GDP（亿元）', 'P值_所在城市一般公共预算支出（万元）',
                         'P值_所在城市教育支出（万元）','P值_所在城市科技支出（万元）','P值_所在城市二产比重']

    for i in range(len(resist_index_list)):
        variable_index = resist_index_list[i]
        significant_index = significant_resist_list[i]

        # Extract parameter estimates and p-values for 'distance' from the GWR model
        param_estimates = gwr_model[variable_index]
        Longitude = gwr_model[Longitude_index]
        Latitude = gwr_model[latitude_index]
        pvalues_distance = gwr_model[significant_index]
        print(pvalues_distance)

        # Find indices where p-value is less than 0.10
        # 找到p值小于0.1的索引
        significant_mask = pvalues_distance < 0.1
        # 提取这些索引对应的经度、纬度和参数估计
        jingdu_coords = Longitude[significant_mask]
        print(jingdu_coords)
        weidu_coords = Latitude[significant_mask]
        significant_params = param_estimates[significant_mask]

        significant_data = pd.DataFrame({
            'Longitude': jingdu_coords,
            'Latitude': weidu_coords,
            'Parameter Value': significant_params,
        })

        print(significant_data)
        print(len(significant_data))
        significant_data.to_csv('significant_data_{}_from_{}.csv'.format(variable_index, sigh_for_significant_data),
                                index=False)

        desc_stats = significant_data.describe()
        desc_stats.to_csv('significant_data_{}_from_{}_城市层面参数描述性统计结果.csv'.format(variable_index, sigh_for_significant_data),
                          index=False)

        '''
            Longitude   Latitude  Parameter Value Start City
        0  117.362499  28.749783        13.563012         上饶
        1  118.310951  35.292000        16.942450         临沂
        2  103.580582  29.174616        20.150308         乐山
        3  105.392094  26.123164        25.343645        六盘水
        4  104.856604  29.614104        19.133537         内江

        '''

        '''创建自定义的红色色条'''

        # 创建一个只有红色渐变的色彩映射
        colors = [(1, 0, 0), (1, 0.95, 0.95)]  # R -> R
        n_bins = 100  # Discretizes the interpolation into bins
        cmap_name = 'custom1'
        red_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        plt.rcParams["font.sans-serif"] = ['Microsoft YaHei']
        '''geopandas 来绘制地图，并根据参数值的大小为每个点分配颜色。'''
        # 创建一个 GeoDataFrame，其中 'geometry' 列包含每个点的坐标
        gdf = gpd.GeoDataFrame(significant_data,
                               geometry=gpd.points_from_xy(significant_data.Longitude, significant_data.Latitude))

        # 中国边界地图
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        china = world[world.name == "China"]

        # 绘制地图
        fig, ax = plt.subplots(figsize=(15, 15))
        china.plot(ax=ax, color='white', edgecolor='black')
        gdf.plot(column='Parameter Value', ax=ax, legend=True, markersize=50, cmap="Reds")
        plt.title("{}的空间异质性".format(variable_index), fontsize=30)
        ax.set_axis_off()
        plt.savefig(f'城市层面的{variable_index}的空间异质性_from_{sigh_for_significant_data}.jpg', dpi=300)
        plt.close()


























'''男性样本'''


'''重新使用在推拉模型中完成的代码进一步提高significant表格和绘图的自动化速度'''

# 读取CSV文件
# 读取CSV文件
file_path_list = ['男性样本GWLR_regression_results_加上城市层面变量之后.csv'
                  ]

for file_path in file_path_list:
    df = pd.read_csv(file_path)
    # print(df.head())
    sigh_for_significant_data = file_path.replace('/', '_')
    print(sigh_for_significant_data)

    select_column = ['P值_年龄段_中年', 'P值_年龄段_壮年', 'P值_年龄段_老年', 'P值_受教育程度_中学学历',
                           'P值_受教育程度_大学专科', 'P值_受教育程度_本科及以上学历', 'P值_户口性质_农业', 'P值_婚姻状况_已婚',
                           'P值_婚姻状况_离婚', 'P值_本次流动原因_家庭类', 'P值_本次流动原因_工作类', 'P值_感觉被看不起_同意',
                     'P值_distance', 'P值_所在城市常住人口（万人）', 'P值_所在城市GDP（亿元）', 'P值_所在城市一般公共预算支出（万元）',
                     'P值_所在城市教育支出（万元）', 'P值_所在城市科技支出（万元）', 'P值_所在城市二产比重'
                     ]
    df_significant = df[select_column]
    # 计算显著性的行数，判断该变量是否有空间研究价值
    count_less_than_01 = (df_significant < 0.1).sum()
    print(count_less_than_01)

    '''开始绘制指定变量的空间异质性'''
    gwr_model = df

    # 去除左右方括号
    gwr_model['城市位置'] = gwr_model['城市位置'].str.replace('[', '').str.replace(']', '')

    ''' 从 GWR 模型中提取参数估计和 p 值:'''
    # Assuming 'distance' is the variable of interest and it's the first column in X
    gwr_model[['Longitude_index', 'latitude_index']] = gwr_model['城市位置'].str.split(r'\s+', expand=True).iloc[:,
                                                       :2].astype(float)
    # print(gwr_model['Longitude_index'].head(300))
    # print(gwr_model['latitude_index'])
    #
    print(gwr_model.head(100))

    ''' 从 GWR 模型中提取参数估计和 p 值:'''
    # Assuming 'distance' is the variable of interest and it's the first column in X
    Longitude_index = 'Longitude_index'
    latitude_index = 'latitude_index'
    variable_index_list = ['参数_年龄段_中年', '参数_年龄段_壮年', '参数_年龄段_老年', '参数_受教育程度_中学学历',
                           '参数_受教育程度_大学专科', '参数_受教育程度_本科及以上学历', '参数_户口性质_农业', '参数_婚姻状况_已婚',
                           '参数_婚姻状况_离婚', '参数_本次流动原因_家庭类', '参数_本次流动原因_工作类', '参数_感觉被看不起_同意']
    significant_index_list = ['P值_年龄段_中年', 'P值_年龄段_壮年', 'P值_年龄段_老年', 'P值_受教育程度_中学学历',
                           'P值_受教育程度_大学专科', 'P值_受教育程度_本科及以上学历', 'P值_户口性质_农业', 'P值_婚姻状况_已婚',
                           'P值_婚姻状况_离婚', 'P值_本次流动原因_家庭类', 'P值_本次流动原因_工作类', 'P值_感觉被看不起_同意']
    resist_index_list = ['参数_distance', '参数_所在城市常住人口（万人）', '参数_所在城市GDP（亿元）', '参数_所在城市一般公共预算支出（万元）',
                         '参数_所在城市教育支出（万元）','参数_所在城市科技支出（万元）','参数_所在城市二产比重']
    significant_resist_list = ['P值_distance', 'P值_所在城市常住人口（万人）', 'P值_所在城市GDP（亿元）', 'P值_所在城市一般公共预算支出（万元）',
                         'P值_所在城市教育支出（万元）','P值_所在城市科技支出（万元）','P值_所在城市二产比重']

    for i in range(len(variable_index_list)):
        variable_index = variable_index_list[i]
        significant_index = significant_index_list[i]

        # Extract parameter estimates and p-values for 'distance' from the GWR model
        param_estimates = gwr_model[variable_index]
        Longitude = gwr_model[Longitude_index]
        Latitude = gwr_model[latitude_index]
        pvalues_distance = gwr_model[significant_index]
        print(pvalues_distance)

        # Find indices where p-value is less than 0.10
        # 找到p值小于0.1的索引
        significant_mask = pvalues_distance < 0.1
        # 提取这些索引对应的经度、纬度和参数估计
        jingdu_coords = Longitude[significant_mask]
        print(jingdu_coords)
        weidu_coords = Latitude[significant_mask]
        significant_params = param_estimates[significant_mask]

        significant_data = pd.DataFrame({
            'Longitude': jingdu_coords,
            'Latitude': weidu_coords,
            'Parameter Value': significant_params,
        })

        print(significant_data)
        print(len(significant_data))
        significant_data.to_csv('significant_data_{}_from_{}.csv'.format(variable_index, sigh_for_significant_data),
                                index=False)

        desc_stats = significant_data.describe()
        desc_stats.to_csv('significant_data_{}_from_{}_参数描述性统计结果.csv'.format(variable_index, sigh_for_significant_data),
                          index=False)

        '''
            Longitude   Latitude  Parameter Value Start City
        0  117.362499  28.749783        13.563012         上饶
        1  118.310951  35.292000        16.942450         临沂
        2  103.580582  29.174616        20.150308         乐山
        3  105.392094  26.123164        25.343645        六盘水
        4  104.856604  29.614104        19.133537         内江

        '''

        '''创建自定义的红色色条'''

        # 创建一个只有红色渐变的色彩映射
        colors = [(1, 0, 0), (1, 0.95, 0.95)]  # R -> R
        n_bins = 100  # Discretizes the interpolation into bins
        cmap_name = 'custom1'
        red_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        plt.rcParams["font.sans-serif"] = ['Microsoft YaHei']
        '''geopandas 来绘制地图，并根据参数值的大小为每个点分配颜色。'''
        # 创建一个 GeoDataFrame，其中 'geometry' 列包含每个点的坐标
        gdf = gpd.GeoDataFrame(significant_data,
                               geometry=gpd.points_from_xy(significant_data.Longitude, significant_data.Latitude))

        # 中国边界地图
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        china = world[world.name == "China"]

        # 绘制地图
        fig, ax = plt.subplots(figsize=(15, 15))
        china.plot(ax=ax, color='white', edgecolor='black')
        gdf.plot(column='Parameter Value', ax=ax, legend=True, markersize=50, cmap="Reds")
        plt.title("{}的空间异质性".format(variable_index), fontsize=30)
        ax.set_axis_off()
        plt.savefig(f'{variable_index}的空间异质性_from_{sigh_for_significant_data}.jpg', dpi=300)
        plt.close()





















'''女性样本'''


'''重新使用在推拉模型中完成的代码进一步提高significant表格和绘图的自动化速度'''

# 读取CSV文件
# 读取CSV文件
file_path_list = ['女性样本GWLR_regression_results_加上城市层面变量之后.csv'
                  ]

for file_path in file_path_list:
    df = pd.read_csv(file_path)
    # print(df.head())
    sigh_for_significant_data = file_path.replace('/', '_')
    print(sigh_for_significant_data)

    select_column = ['P值_年龄段_中年', 'P值_年龄段_壮年', 'P值_年龄段_老年', 'P值_受教育程度_中学学历',
                           'P值_受教育程度_大学专科', 'P值_受教育程度_本科及以上学历', 'P值_户口性质_农业', 'P值_婚姻状况_已婚',
                           'P值_婚姻状况_离婚', 'P值_本次流动原因_家庭类', 'P值_本次流动原因_工作类', 'P值_感觉被看不起_同意',
                     'P值_distance', 'P值_所在城市常住人口（万人）', 'P值_所在城市GDP（亿元）', 'P值_所在城市一般公共预算支出（万元）',
                     'P值_所在城市教育支出（万元）', 'P值_所在城市科技支出（万元）', 'P值_所在城市二产比重'
                     ]
    df_significant = df[select_column]
    # 计算显著性的行数，判断该变量是否有空间研究价值
    count_less_than_01 = (df_significant < 0.1).sum()
    print(count_less_than_01)

    '''开始绘制指定变量的空间异质性'''
    gwr_model = df

    # 去除左右方括号
    gwr_model['城市位置'] = gwr_model['城市位置'].str.replace('[', '').str.replace(']', '')

    ''' 从 GWR 模型中提取参数估计和 p 值:'''
    # Assuming 'distance' is the variable of interest and it's the first column in X
    gwr_model[['Longitude_index', 'latitude_index']] = gwr_model['城市位置'].str.split(r'\s+', expand=True).iloc[:,
                                                       :2].astype(float)
    # print(gwr_model['Longitude_index'].head(300))
    # print(gwr_model['latitude_index'])
    #
    print(gwr_model.head(100))

    ''' 从 GWR 模型中提取参数估计和 p 值:'''
    # Assuming 'distance' is the variable of interest and it's the first column in X
    Longitude_index = 'Longitude_index'
    latitude_index = 'latitude_index'
    variable_index_list = ['参数_年龄段_中年', '参数_年龄段_壮年', '参数_年龄段_老年', '参数_受教育程度_中学学历',
                           '参数_受教育程度_大学专科', '参数_受教育程度_本科及以上学历', '参数_户口性质_农业', '参数_婚姻状况_已婚',
                           '参数_婚姻状况_离婚', '参数_本次流动原因_家庭类', '参数_本次流动原因_工作类', '参数_感觉被看不起_同意']
    significant_index_list = ['P值_年龄段_中年', 'P值_年龄段_壮年', 'P值_年龄段_老年', 'P值_受教育程度_中学学历',
                           'P值_受教育程度_大学专科', 'P值_受教育程度_本科及以上学历', 'P值_户口性质_农业', 'P值_婚姻状况_已婚',
                           'P值_婚姻状况_离婚', 'P值_本次流动原因_家庭类', 'P值_本次流动原因_工作类', 'P值_感觉被看不起_同意']
    resist_index_list = ['参数_distance', '参数_所在城市常住人口（万人）', '参数_所在城市GDP（亿元）', '参数_所在城市一般公共预算支出（万元）',
                         '参数_所在城市教育支出（万元）','参数_所在城市科技支出（万元）','参数_所在城市二产比重']
    significant_resist_list = ['P值_distance', 'P值_所在城市常住人口（万人）', 'P值_所在城市GDP（亿元）', 'P值_所在城市一般公共预算支出（万元）',
                         'P值_所在城市教育支出（万元）','P值_所在城市科技支出（万元）','P值_所在城市二产比重']

    for i in range(len(resist_index_list)):
        variable_index = resist_index_list[i]
        significant_index = significant_resist_list[i]

        # Extract parameter estimates and p-values for 'distance' from the GWR model
        param_estimates = gwr_model[variable_index]
        Longitude = gwr_model[Longitude_index]
        Latitude = gwr_model[latitude_index]
        pvalues_distance = gwr_model[significant_index]
        print(pvalues_distance)

        # Find indices where p-value is less than 0.10
        # 找到p值小于0.1的索引
        significant_mask = pvalues_distance < 0.1
        # 提取这些索引对应的经度、纬度和参数估计
        jingdu_coords = Longitude[significant_mask]
        print(jingdu_coords)
        weidu_coords = Latitude[significant_mask]
        significant_params = param_estimates[significant_mask]

        significant_data = pd.DataFrame({
            'Longitude': jingdu_coords,
            'Latitude': weidu_coords,
            'Parameter Value': significant_params,
        })

        print(significant_data)
        print(len(significant_data))
        significant_data.to_csv('significant_data_{}_from_{}.csv'.format(variable_index, sigh_for_significant_data),
                                index=False)

        desc_stats = significant_data.describe()
        desc_stats.to_csv('significant_data_{}_from_{}_参数描述性统计结果.csv'.format(variable_index, sigh_for_significant_data),
                          index=False)

        '''
            Longitude   Latitude  Parameter Value Start City
        0  117.362499  28.749783        13.563012         上饶
        1  118.310951  35.292000        16.942450         临沂
        2  103.580582  29.174616        20.150308         乐山
        3  105.392094  26.123164        25.343645        六盘水
        4  104.856604  29.614104        19.133537         内江

        '''

        '''创建自定义的红色色条'''

        # 创建一个只有红色渐变的色彩映射
        colors = [(1, 0, 0), (1, 0.95, 0.95)]  # R -> R
        n_bins = 100  # Discretizes the interpolation into bins
        cmap_name = 'custom1'
        red_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        plt.rcParams["font.sans-serif"] = ['Microsoft YaHei']
        '''geopandas 来绘制地图，并根据参数值的大小为每个点分配颜色。'''
        # 创建一个 GeoDataFrame，其中 'geometry' 列包含每个点的坐标
        gdf = gpd.GeoDataFrame(significant_data,
                               geometry=gpd.points_from_xy(significant_data.Longitude, significant_data.Latitude))

        # 中国边界地图
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        china = world[world.name == "China"]

        # 绘制地图
        fig, ax = plt.subplots(figsize=(15, 15))
        china.plot(ax=ax, color='white', edgecolor='black')
        gdf.plot(column='Parameter Value', ax=ax, legend=True, markersize=50, cmap="Reds")
        plt.title("{}的空间异质性".format(variable_index), fontsize=30)
        ax.set_axis_off()
        plt.savefig(f'{variable_index}的空间异质性_from_{sigh_for_significant_data}.jpg', dpi=300)
        plt.close()
