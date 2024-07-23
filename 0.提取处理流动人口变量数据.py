import pandas as pd
import numpy as np
# 读取Excel文件
gaode_data = pd.read_csv("2018流动人口动态监测.csv")
print(gaode_data.head())
print(len(gaode_data))

selected_columns = ['newID','C2','C3','Q313','Q314','Q315','q101a1','q101b1','q101c1y','q101c1m','q101m1y',
                    'q101d1','q101e1','q101f1','q101h1',
                    'q101i1','q101k1','q101j1a','q101j1b','q101l1','q101n1','Q109','Q201A',
                    'Q503A','Q503D','Q503E','Q503H','Q215','Q216','Q311D','Q311E',
                    'Q100','hjpro','Q103','Q104','Q105','Q301']


selected_df = gaode_data[selected_columns]
print(selected_df)

selected_df = selected_df[selected_columns]
selected_df.to_csv('抽取的迁徙调查数据0313.csv',index=False)



'''映射为标签'''

# 读取 "抽取的迁徙调查数据.csv" 的前几行数据以查看其结构
migration_data_sample = pd.read_csv('抽取的迁徙调查数据0313.csv', nrows=5)

# 显示前几行数据
print(migration_data_sample.head())
# 读取 "变量名称.xlsx" 的数据以查看其结构
variable_mapping = pd.read_excel('变量名称.xlsx')

# 显示变量名称映射信息
print(variable_mapping.head())

# 读取完整的 "抽取的迁徙调查数据.csv" 数据
migration_data = pd.read_csv('抽取的迁徙调查数据0313.csv')

# 创建一个字典用于存储 "Variable" 到 "Label" 的映射关系
variable_to_label_mapping = dict(zip(variable_mapping['Variable'], variable_mapping['Label']))

# 替换原数据的列名
new_columns = [variable_to_label_mapping.get(col, col) for col in migration_data.columns]
migration_data.columns = new_columns

# 保存替换后的数据到新的CSV文件
new_csv_path = '抽取的迁徙调查数据_替换后0313.csv'
migration_data.to_csv(new_csv_path, index=False)

# 显示替换后的数据的前几行以进行检查
migration_data.head(), new_csv_path


'''提取独立的流动城市并输出复制两列，左列原，右列联通标准'''
# 读取 '筛选后的行政区划代码_无省.csv' 文件
admin_code_path = '筛选后的行政区划代码_无省.csv'
admin_code_df = pd.read_csv(admin_code_path,encoding='GBK')
print(admin_code_df.head())

# 读取 'GPT_代码转城市.xlsx' 文件
code_to_city_path = '抽取的迁徙调查数据_替换后0313.csv'
code_to_city_df = pd.read_csv(code_to_city_path)
print(code_to_city_df.head())
print(len(code_to_city_df))

# 提取 '行政区划代码' 和 '单位名称' 列，并将 '行政区划代码' 的前四位作为新列
admin_code_df['前四位行政区划代码'] = admin_code_df['行政区划代码'].astype(str).str[:4]
# 提取 '户籍地区县行政区划代码' 列，并将其前四位作为新列
code_to_city_df['前四位行政区划代码'] = code_to_city_df['户籍地区县行政区划代码           '].astype(str).str[:4]

# 根据 '前四位行政区划代码' 进行匹配，生成一个新列以存储匹配上的 '单位名称'
merged_df = pd.merge(code_to_city_df, admin_code_df[['前四位行政区划代码', '单位名称']], on='前四位行政区划代码', how='left')

# 显示合并后的 DataFrame 的前几行，以确认匹配结果
merged_df.head(50)

# 保存更新后的 DataFrame 为新的 Excel 文件
output_path = '流动检测数据_生成了老家城市的联通列0313.csv'
merged_df.to_csv(output_path, index=False)



'''map一次，先map成标准联通的，然后再map成高德的'''
city_mapping_df = pd.read_csv("流动检测数据_生成了老家城市的联通列0313.csv",encoding='GBK')
print(city_mapping_df.head())
df1 = pd.read_csv('校正后的城市和收入（用于合并）.csv',encoding='GBK')
print(df1.head())

city_mapping_df['city'] = city_mapping_df['老家城市'].map(df1.set_index('联通出发到达市')['startCity'])
print(city_mapping_df.head(100))

'''添加高德经纬度'''
cities_df = pd.read_csv("高德数据完整城市经纬度.csv", encoding="GBK")
print(cities_df.head())
# Attempt the merge operation again after reloading
base_model = city_mapping_df.merge(cities_df[['city', 'City_longitude', 'City_latitude']],
                                          on='city',
                                          how='left')
base_model.rename(columns={'City_longitude': '老家经度', 'City_latitude': '老家纬度'}, inplace=True)
print(base_model.head(100))

base_model.to_csv('流动检测数据_生成了老家城市的联通列+经纬度0313.csv', index=False)


'''查看所有独立的流动城市并输出'''
df = pd.read_csv("流动检测数据_生成了老家城市的联通列+经纬度0313.csv" )
print(df.head())

city_list = df['现居住地址市（地区）          '].unique()
print(city_list,len(city_list))

# 将列表转换为DataFrame
df = pd.DataFrame({'流动监测的当地城市': city_list})
# 将DataFrame输出为CSV文件
df.to_csv('流动监测的当地城市（用于比较联通）.csv', index=False)
print("DataFrame已保存为CSV文件")




'''映射一'''
df = pd.read_csv("流动监测的当地城市（用于比较联通）.csv",encoding='GBK')
print(df.head())


city_mapping_df = pd.read_csv("流动检测数据_生成了老家城市的联通列+经纬度0313.csv")
print(city_mapping_df.head())

unique_df = df.drop_duplicates(subset=['流动监测的当地城市'], keep='first')
print(unique_df)
city_mapping_df['city_current'] = city_mapping_df['现居住地址市（地区）          '].map(unique_df.set_index('流动监测的当地城市')['流动监测的当地城市映射'])
print(city_mapping_df.head(1000))


'''映射二'''
'''map一次，先map成标准联通的，然后再map成高德的'''
df1 = pd.read_csv('校正后的城市和收入（用于合并）.csv',encoding='GBK')
print(df1.head())

city_mapping_df['city_current_gaode'] = city_mapping_df['city_current'].map(df1.set_index('联通出发到达市')['startCity'])
print(city_mapping_df.head(100))
city_mapping_df.to_csv('流动检测数据_生成了老家城市的联通列+经纬度+当地城市的高德列0314.csv', index=False)


'''添加高德经纬度'''
city_mapping_df = pd.read_csv('流动检测数据_生成了老家城市的联通列+经纬度+当地城市的高德列0314.csv', encoding="GBK")
cities_df = pd.read_csv("高德数据完整城市经纬度.csv", encoding="GBK")
print(cities_df.head())
# Attempt the merge operation again after reloading
base_model = city_mapping_df.merge(cities_df[['city', 'City_longitude', 'City_latitude']],
                                          on='city',
                                          how='left')
base_model.rename(columns={'City_longitude': '当地经度', 'City_latitude': '当地纬度'}, inplace=True)
print(base_model.head(100))

base_model.to_csv('流动检测数据_生成了老家城市的联通列+经纬度+当地城市的高德列+经纬度0314.csv', index=False)


'''去除经纬度缺失的行，然后生成距离'''
base_model = pd.read_csv('流动检测数据_生成了老家城市的联通列+经纬度+当地城市的高德列+经纬度0314.csv')
print(len(base_model))
base_model = base_model.dropna(subset=['当地经度','当地纬度','老家经度','老家纬度']).reset_index(drop=True)
print(len(base_model))


# 定义函数计算两点之间的距离
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # 地球平均半径，单位为公里
    return c * r

# 计算startCity和endCity之间的空间距离，并添加到新列distance中
base_model['distance'] = base_model.apply(lambda row: haversine(row['当地经度'],
                                                                           row['当地纬度'],
                                                                           row['老家经度'],
                                                                           row['老家纬度']), axis=1)

print(base_model.head(500))

base_model.to_csv('流动检测数据_生成了老家城市的联通列+经纬度+当地城市的高德列+经纬度+距离0314.csv', index=False)


'''对感兴趣的列提取独立值，然后保留需要的行'''
base_model = pd.read_csv('流动检测数据_生成了老家城市的联通列+经纬度+当地城市的高德列+经纬度+距离0314.csv')
print(base_model.head())
print(len(base_model))
print(base_model['现居住地址市（地区）          '].unique(),len(base_model['现居住地址市（地区）          '].unique()))
print(base_model['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].unique())
df_filtered = base_model[base_model['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].isin(['愿意','不愿意'])]
print(len(df_filtered))
df_filtered_reset = df_filtered.reset_index(drop=True)
print(df_filtered_reset)

print(df_filtered_reset['与被访者关系      '].unique())

print(df_filtered_reset['性别  '].unique())

print(df_filtered_reset['受教育程度     '].unique())
# 替换特定的值
df_filtered_reset['受教育程度     '].replace(['未上过小学', '小学'], '小学及以下学历', inplace=True)
# 重置索引，并从0开始
df_filtered_reset['受教育程度     '].replace(['初中', '高中/中专'], '中学学历', inplace=True)
df_filtered_reset['受教育程度     '].replace(['大学专科','大学本科','研究生'], '大学及以上学历', inplace=True)


print(df_filtered_reset['户口性质    '].unique())

df_filtered_reset = df_filtered_reset[df_filtered_reset['户口性质    '].isin(['农业','非农业'])]
df_filtered_reset = df_filtered_reset.reset_index(drop=True)
print(len(df_filtered_reset))
print(df_filtered_reset)


print(df_filtered_reset['婚姻状况    '].unique())
df_filtered_reset = df_filtered_reset[df_filtered_reset['婚姻状况    '].isin(['未婚','初婚','再婚'])]
print(len(df_filtered_reset))

df_filtered_reset['婚姻状况    '].replace(['初婚','再婚'], '已婚', inplace=True)


print(df_filtered_reset['是否本地户籍人口        '].unique())

print(df_filtered_reset['本次流动原因      '].unique())
#
# df_filtered_reset = df_filtered_reset[df_filtered_reset['本次流动原因      '].isin(['经商','务工/工作','家属随迁','婚姻嫁娶','照顾自家小孩','照顾自家老人'])]
# df_filtered_reset['本次流动原因      '].replace(['务工/工作'], '工作类', inplace=True)
# df_filtered_reset['本次流动原因      '].replace(['经商'], '经商类', inplace=True)
# df_filtered_reset['本次流动原因      '].replace(['家属随迁','婚姻嫁娶','照顾自家小孩','照顾自家老人'], '家庭类', inplace=True)

print(df_filtered_reset['您是否同意“我感觉本地人看不起外地人”这个说法                         '].unique())
df_filtered_reset['您是否同意“我感觉本地人看不起外地人”这个说法                         '].replace(['基本同意','完全同意'], '同意', inplace=True)
df_filtered_reset['您是否同意“我感觉本地人看不起外地人”这个说法                         '].replace(['完全不同意','不同意'], '不同意', inplace=True)

# 添加新加的一些变量
#
# print(df_filtered_reset['您户籍地老家是否有宅基地            '].unique())
# df_filtered_reset = df_filtered_reset[df_filtered_reset['您户籍地老家是否有宅基地            '].isin(['没有','有'])]
# print(len(df_filtered_reset))


# print(df_filtered_reset['同住的家庭成员人数         '].unique())
df_filtered_reset['出生年'] = df_filtered_reset['出生年   '].astype(int)
df_filtered_reset['年龄'] = 2018 - df_filtered_reset['出生年']
print(df_filtered_reset)


df_filtered_reset['流动年份'] = df_filtered_reset['本次流动年份      '].astype(int)
df_filtered_reset['流动时长'] = 2018 - df_filtered_reset['流动年份']
print(df_filtered_reset)



df_filtered_reset.to_csv('跑模型之前的整理（尚未添加虚拟变量）0314.csv', index=False)


'''二轮修改，添加虚拟变量等'''
base_model = pd.read_csv('跑模型之前的整理（尚未添加虚拟变量）0314.csv')
print(base_model.head())
print(len(base_model))

print(base_model['年龄'].unique(), max(base_model['年龄'].unique()),min(base_model['年龄'].unique()))

base_model['年龄'] = base_model['年龄'].astype(int)
base_model['年龄段'] = ''

base_model.loc[(base_model['年龄'] >= 18) & (base_model['年龄'] < 40), '年龄段'] = '青年'
base_model.loc[(base_model['年龄'] >= 40) & (base_model['年龄'] < 60), '年龄段'] = '中年'
base_model.loc[(base_model['年龄'] >= 60), '年龄段'] = '老年'

print(base_model['年龄段'].unique())
base_model = base_model[base_model['年龄段'] != '']
print(len(base_model))
# 指定要保留的列
columns_to_keep = ['现居住地址市（地区）          ', '如果您符合本地落户条件，您是否愿意把户口迁入本地                        ',
                   '性别  ','受教育程度     ','户口性质    ','婚姻状况    ','年龄段',
                   '本次流动范围      ','流动时长','同住的家庭成员人数         ','您是否同意“我感觉本地人看不起外地人”这个说法                         ',
                   '老家城市','city_current','当地经度','当地纬度','distance']

df_new = base_model[columns_to_keep]
print(len(df_new))
columns_to_check = ['如果您符合本地落户条件，您是否愿意把户口迁入本地                        ', '性别  ','年龄段',
                    '受教育程度     ','户口性质    ','婚姻状况    ','本次流动范围      ','流动时长','同住的家庭成员人数         ','您是否同意“我感觉本地人看不起外地人”这个说法                         ',
                   '老家城市','city_current','当地经度','当地纬度','distance']
df_new = df_new.dropna(subset=columns_to_check).reset_index(drop=True)
print(df_new)

# 开始加虚拟变量
df_new['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].replace(['愿意'], 1, inplace=True)
df_new['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].replace(['不愿意'], 0, inplace=True)

# 生成虚拟变量
dummies = pd.get_dummies(df_new['性别  '], prefix='性别').astype(int)

# 删除"男"这一列，以其为基准
dummies.drop('性别_男', axis=1, inplace=True)
print(dummies)

# 将生成的虚拟变量添加到原始DataFrame中
df_new = pd.concat([df_new, dummies], axis=1)
print(df_new)


# 生成虚拟变量
dummies = pd.get_dummies(df_new['年龄段'], prefix='年龄段').astype(int)

# 删除"男"这一列，以其为基准
dummies.drop('年龄段_老年', axis=1, inplace=True)
print(dummies)

# 将生成的虚拟变量添加到原始DataFrame中
df_new = pd.concat([df_new, dummies], axis=1)





# 生成虚拟变量
dummies = pd.get_dummies(df_new['受教育程度     '], prefix='受教育程度').astype(int)
print(dummies)
# 删除"男"这一列，以其为基准
dummies.drop('受教育程度_小学及以下学历', axis=1, inplace=True)
print(dummies)

# 将生成的虚拟变量添加到原始DataFrame中
df_new = pd.concat([df_new, dummies], axis=1)






print(df_new['户口性质    '].unique())
# 生成虚拟变量
dummies = pd.get_dummies(df_new['户口性质    '], prefix='户口性质').astype(int)
print(dummies)
# 删除"男"这一列，以其为基准
dummies.drop('户口性质_非农业', axis=1, inplace=True)
print(dummies)

# 将生成的虚拟变量添加到原始DataFrame中
df_new = pd.concat([df_new, dummies], axis=1)






# 生成虚拟变量
dummies = pd.get_dummies(df_new['婚姻状况    '], prefix='婚姻状况').astype(int)

# 删除"男"这一列，以其为基准
dummies.drop('婚姻状况_未婚', axis=1, inplace=True)
print(dummies)

# 将生成的虚拟变量添加到原始DataFrame中
df_new = pd.concat([df_new, dummies], axis=1)




#
# # 生成虚拟变量
# dummies = pd.get_dummies(df_new['本次流动原因      '], prefix='本次流动原因').astype(int)
#
# # 删除"男"这一列，以其为基准
# dummies.drop('本次流动原因_经商类', axis=1, inplace=True)
# print(dummies)
#
# # 将生成的虚拟变量添加到原始DataFrame中
# df_new = pd.concat([df_new, dummies], axis=1)



# 生成虚拟变量
dummies = pd.get_dummies(df_new['您是否同意“我感觉本地人看不起外地人”这个说法                         '], prefix='感觉被看不起').astype(int)

# 删除"男"这一列，以其为基准
dummies.drop('感觉被看不起_不同意', axis=1, inplace=True)
print(dummies)

# 将生成的虚拟变量添加到原始DataFrame中
df_new = pd.concat([df_new, dummies], axis=1)

print(df_new.head())



# 添加一个跨越空间的虚拟变量

# 生成虚拟变量
dummies = pd.get_dummies(df_new['本次流动范围      '], prefix='跨越尺度').astype(int)

# 删除"男"这一列，以其为基准
dummies.drop('跨越尺度_市内跨县', axis=1, inplace=True)
print(dummies)

# 将生成的虚拟变量添加到原始DataFrame中
df_new = pd.concat([df_new, dummies], axis=1)

print(df_new.head())





print(len(df_new))
df_new.to_csv('logit模型数据表（不包含性别交叉项）0314.csv', index=False)