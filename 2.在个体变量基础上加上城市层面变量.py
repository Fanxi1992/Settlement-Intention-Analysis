import matplotlib

matplotlib.use('TkAgg')  # 设置为Agg后端
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay

import pandas as pd
import numpy as np


'''
#merge一些闲鱼上的指标，merge之前核对原始城市列和闲鱼数据的城市列，有不对应的，就改闲鱼那张表
statistics = pd.read_excel("3.指标多 1998～2020年中国城市统计年鉴地级市面板数据.xlsx")
print(statistics.head(100))
df_2020 = statistics[statistics['年份'] == 2020]
print(len(df_2020))
print(df_2020.head(10))
selected_columns = df_2020[['年份', '市','年末户籍人口_万人_全市','常住人口_万人_全市']]
selected_columns.to_csv('常住人口数据2020（供匹配联通）.csv', index=False)
'''

base_model = pd.read_csv('logit模型数据表（不包含性别交叉项）0314.csv',encoding='GBK')
print(base_model.head())

population = pd.read_csv("常住人口数据2020（供匹配联通）.csv",encoding='GBK')
print(population.head())


base_model = base_model.merge(population[['出发市', '常住人口_万人_全市']],
                                          on='出发市',
                                          how='left')
base_model.rename(columns={'常住人口_万人_全市': '所在城市常住人口（万人）'}, inplace=True)

print(base_model.head(100))








# 开始处理2017-2018年的数据
city_factors = pd.read_excel('1.中国城市统计面板数据2000-2021年.xlsx')
print(city_factors)

df_2018 = city_factors[city_factors['年份'] == 2018]
print(len(df_2018))
df_2018.reset_index(drop=True)

columns_to_keep = ['城市','地区生产总值_当年价格-全市（亿元）', '户籍人口-全市(万人)', '在岗职工平均工资_元_全市', '第三产业占地区生产总值的比重-全市', '城镇登记失业人员数-全市(人)']
df_filtered = df_2018[columns_to_keep]
df_filtered.to_csv('2018年城市层面数据.csv', index=False)




other_city_factors = pd.read_csv('2018年城市层面数据.csv',encoding='GBK')
print(other_city_factors.head(15))
print(len(other_city_factors))

other_city_factors['人均GDP_万元'] = other_city_factors['地区生产总值_当年价格-全市（亿元）'] / other_city_factors['户籍人口-全市(万人)']
other_city_factors['失业率'] = other_city_factors['城镇登记失业人员数-全市(人)'] / (10000*other_city_factors['户籍人口-全市(万人)'])


base_model = base_model.merge(other_city_factors[['出发市', '地区生产总值_当年价格-全市（亿元）']],
                                          on='出发市',
                                          how='left')
base_model.rename(columns={'地区生产总值_当年价格-全市（亿元）': '所在城市GDP_亿元'}, inplace=True)

print(base_model.head(100))



base_model = base_model.merge(other_city_factors[['出发市', '在岗职工平均工资_元_全市']],
                                          on='出发市',
                                          how='left')
base_model.rename(columns={'在岗职工平均工资_元_全市': '所在城市在岗职工平均工资_元'}, inplace=True)

print(base_model.head(100))



base_model = base_model.merge(other_city_factors[['出发市', '第三产业占地区生产总值的比重-全市']],
                                          on='出发市',
                                          how='left')
base_model.rename(columns={'第三产业占地区生产总值的比重-全市': '所在城市第三产业比重'}, inplace=True)

print(base_model.head(100))



#
# #---------------
# base_model = base_model.merge(other_city_factors[['出发市', '地方一般公共预算支出（万元）全市']],
#                                           on='出发市',
#                                           how='left')
# base_model.rename(columns={'地方一般公共预算支出（万元）全市': '所在城市一般公共预算支出（万元）'}, inplace=True)
#
# print(base_model.head(100))
#
# #------------------------------
# base_model = base_model.merge(other_city_factors[['出发市', '教育支出（万元）全市']],
#                                           on='出发市',
#                                           how='left')
# base_model.rename(columns={'教育支出（万元）全市': '所在城市教育支出（万元）'}, inplace=True)
#
# print(base_model.head(100))
#
# #----------------------------------
# base_model = base_model.merge(other_city_factors[['出发市', '科学技术支出（万元）全市']],
#                                           on='出发市',
#                                           how='left')
# base_model.rename(columns={'科学技术支出（万元）全市': '所在城市科技支出（万元）'}, inplace=True)
#
# print(base_model.head(100))


'''做个VIF'''
'''
from statsmodels.stats.outliers_influence import variance_inflation_factor
# 假设 df 是你的 DataFrame

# 选择自变量
base_model = pd.read_csv('除了空气质量.csv')
X = base_model[['distance', '到达市常住人口', '出发市常住人口', '到达市GDP', '出发市GDP']].copy()  # 替换为你的实际列名
X = pd.get_dummies(X, drop_first=True)  # 如果有分类变量，进行哑变量编码
X = X.dropna()
# 添加截距
X['intercept'] = 1

# 计算VIF
vif_data = pd.DataFrame()
vif_data["variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)
'''


# 拼接环境因素之前需要先映射为高德城市名
df1 = pd.read_csv('校正后的城市和收入（用于合并）.csv',encoding='GBK')
print(df1.head())

base_model['city_current_gaode'] = base_model['出发市'].map(df1.set_index('联通出发到达市')['startCity'])
print(base_model.head(100))


AIR_quality = pd.read_csv('city_monthly_avg_aqi_pm25_2019_stripped.csv',encoding='GBK')
print(base_model.head())
print(AIR_quality.head())

# base_model = base_model.merge(AIR_quality[['city_current_gaode', '月均AQI']],
#                                           on='city_current_gaode',
#                                           how='left')
# base_model.rename(columns={'月均AQI': '所在城市AQI'}, inplace=True)
#
# print(base_model.head(500))


base_model = base_model.merge(AIR_quality[['city_current_gaode', '月均PM2.5']],
                                          on='city_current_gaode',
                                          how='left')
base_model.rename(columns={'月均PM2.5': '所在城市PM2.5'}, inplace=True)

print(base_model.head(100))
print(len(base_model))

#
# other_city_factors = pd.read_excel('cities_data_2019.xlsx')
# print(other_city_factors.head(15))
# print(len(other_city_factors))
#
# base_model = base_model.merge(other_city_factors[['出发市', '第二产业占地区生产总值的比重-全市']],
#                                           on='出发市',
#                                           how='left')
# base_model.rename(columns={'第二产业占地区生产总值的比重-全市': '所在城市二产比重'}, inplace=True)
#
#
# print(base_model.head(100))
# print(len(base_model))

base_model.to_csv('logit模型数据表（不包含性别交叉项)+城市层面0314.csv', index=False)

base_model = base_model.dropna().reset_index()
print(base_model.head(100))
unique_city_num = base_model['出发市'].unique()
print(len(unique_city_num))
