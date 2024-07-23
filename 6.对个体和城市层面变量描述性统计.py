import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# 假设你的数据存储在'your_data.csv'这个文件中
df = pd.read_csv('logit模型数据表（不包含性别交叉项)+城市层面+去空值并标准化.csv')
print(df.head())

'''分出受教育程度的子样本'''
df_select = df[(df['受教育程度_中学学历'] == 0) & (df['受教育程度_大学专科'] == 0) & (df['受教育程度_本科及以上学历'] == 0)]

selected_rows_copy = df_select.copy()
selected_rows_copy['受教育程度'] = '小学及以下学历'
print(selected_rows_copy.head())
# 为了简化代码，这里我们只处理受教育程度为例，你可以参考这个代码处理其他变量
edu_crosstab = pd.crosstab(selected_rows_copy['受教育程度'], selected_rows_copy['性别  '], margins=True, margins_name="Total")
print(edu_crosstab)
print(edu_crosstab.iloc[:-1, :-1])

'''分出受教育程度的子样本'''
df_select = df[(df['受教育程度_中学学历'] == 1) & (df['受教育程度_大学专科'] == 0) & (df['受教育程度_本科及以上学历'] == 0)]

selected_rows_copy = df_select.copy()
selected_rows_copy['受教育程度'] = '中学学历'
print(selected_rows_copy.head())
# 为了简化代码，这里我们只处理受教育程度为例，你可以参考这个代码处理其他变量
edu_crosstab = pd.crosstab(selected_rows_copy['受教育程度'], selected_rows_copy['性别  '], margins=True, margins_name="Total")
print(edu_crosstab)
print(edu_crosstab.iloc[:-1, :-1])

'''分出受教育程度的子样本'''
df_select = df[(df['受教育程度_中学学历'] == 0) & (df['受教育程度_大学专科'] == 1) & (df['受教育程度_本科及以上学历'] == 0)]

selected_rows_copy = df_select.copy()
selected_rows_copy['受教育程度'] = '大学专科'
print(selected_rows_copy.head())
# 为了简化代码，这里我们只处理受教育程度为例，你可以参考这个代码处理其他变量
edu_crosstab = pd.crosstab(selected_rows_copy['受教育程度'], selected_rows_copy['性别  '], margins=True, margins_name="Total")
print(edu_crosstab)
print(edu_crosstab.iloc[:-1, :-1])

'''分出受教育程度的子样本'''
df_select = df[(df['受教育程度_中学学历'] == 0) & (df['受教育程度_大学专科'] == 0) & (df['受教育程度_本科及以上学历'] == 1)]

selected_rows_copy = df_select.copy()
selected_rows_copy['受教育程度'] = '本科及以上'
print(selected_rows_copy.head())
# 为了简化代码，这里我们只处理受教育程度为例，你可以参考这个代码处理其他变量
edu_crosstab = pd.crosstab(selected_rows_copy['受教育程度'], selected_rows_copy['性别  '], margins=True, margins_name="Total")
print(edu_crosstab)


# 手动输入数据
# 数据结构是：[男性数据], [女性数据]
data = np.array([
    [6554, 8259],  # 非饮酒者
    [33558, 27476],   # 现饮酒者
    [4655, 5161],     # 曾饮酒者
    [3354, 3340]     # 曾饮酒者
])

# 使用chi2_contingency进行卡方检验
chi2, p, _, _ = chi2_contingency(data)

print(f"卡方值为: {chi2:.2f}")
print(f"P值为: {p:.4f}")


'''分出hukou的子样本'''
df_select = df[df['户口性质_农业'] == 0]

selected_rows_copy = df_select.copy()
selected_rows_copy['hukou'] = '农业'
print(selected_rows_copy.head())
# 为了简化代码，这里我们只处理受教育程度为例，你可以参考这个代码处理其他变量
edu_crosstab = pd.crosstab(selected_rows_copy['hukou'], selected_rows_copy['性别  '], margins=True, margins_name="Total")
print(edu_crosstab)



df_select = df[df['户口性质_农业'] == 1]

selected_rows_copy = df_select.copy()
selected_rows_copy['hukou'] = '农业'
print(selected_rows_copy.head())
# 为了简化代码，这里我们只处理受教育程度为例，你可以参考这个代码处理其他变量
edu_crosstab = pd.crosstab(selected_rows_copy['hukou'], selected_rows_copy['性别  '], margins=True, margins_name="Total")
print(edu_crosstab)



# 手动输入数据
# 数据结构是：[男性数据], [女性数据]
data = np.array([
    [40714, 37094],  # 非饮酒者
    [7407, 7142],   # 现饮酒者
])

# 使用chi2_contingency进行卡方检验
chi2, p, _, _ = chi2_contingency(data)

print(f"卡方值为: {chi2:.2f}")
print(f"P值为: {p:.4f}")





'''分出婚姻的子样本'''
df_select = df[(df['婚姻状况_已婚'] == 0) & (df['婚姻状况_离婚'] == 0)]

selected_rows_copy = df_select.copy()
selected_rows_copy['婚姻情况'] = '未婚'
print(selected_rows_copy.head())
# 为了简化代码，这里我们只处理受教育程度为例，你可以参考这个代码处理其他变量
edu_crosstab = pd.crosstab(selected_rows_copy['婚姻情况'], selected_rows_copy['性别  '], margins=True, margins_name="Total")
print(edu_crosstab)

df_select = df[(df['婚姻状况_已婚'] == 1) & (df['婚姻状况_离婚'] == 0)]

selected_rows_copy = df_select.copy()
selected_rows_copy['婚姻情况'] = '已婚'
print(selected_rows_copy.head())
# 为了简化代码，这里我们只处理受教育程度为例，你可以参考这个代码处理其他变量
edu_crosstab = pd.crosstab(selected_rows_copy['婚姻情况'], selected_rows_copy['性别  '], margins=True, margins_name="Total")
print(edu_crosstab)



df_select = df[(df['婚姻状况_已婚'] == 0) & (df['婚姻状况_离婚'] == 1)]

selected_rows_copy = df_select.copy()
selected_rows_copy['婚姻情况'] = 'L婚'
print(selected_rows_copy.head())
# 为了简化代码，这里我们只处理受教育程度为例，你可以参考这个代码处理其他变量
edu_crosstab = pd.crosstab(selected_rows_copy['婚姻情况'], selected_rows_copy['性别  '], margins=True, margins_name="Total")
print(edu_crosstab)


# 手动输入数据
# 数据结构是：[男性数据], [女性数据]
data = np.array([
    [7048, 5853],  # 非饮酒者
    [40272, 37564],   # 现饮酒者
[801, 819],
])

# 使用chi2_contingency进行卡方检验
chi2, p, _, _ = chi2_contingency(data)

print(f"卡方值为: {chi2:.2f}")
print(f"P值为: {p:.4f}")





'''分出年龄的子样本'''
df_select = df[(df['年龄段_中年'] == 0) & (df['年龄段_壮年'] == 0) & (df['年龄段_老年'] == 0)]

selected_rows_copy = df_select.copy()
selected_rows_copy['年龄111'] = '青年'
print(selected_rows_copy.head())
# 为了简化代码，这里我们只处理受教育程度为例，你可以参考这个代码处理其他变量
edu_crosstab = pd.crosstab(selected_rows_copy['年龄111'], selected_rows_copy['性别  '], margins=True, margins_name="Total")
print(edu_crosstab)


df_select = df[(df['年龄段_中年'] == 1) & (df['年龄段_壮年'] == 0) & (df['年龄段_老年'] == 0)]

selected_rows_copy = df_select.copy()
selected_rows_copy['年龄111'] = '中年'
print(selected_rows_copy.head())
# 为了简化代码，这里我们只处理受教育程度为例，你可以参考这个代码处理其他变量
edu_crosstab = pd.crosstab(selected_rows_copy['年龄111'], selected_rows_copy['性别  '], margins=True, margins_name="Total")
print(edu_crosstab)


df_select = df[(df['年龄段_中年'] == 0) & (df['年龄段_壮年'] == 1) & (df['年龄段_老年'] == 0)]

selected_rows_copy = df_select.copy()
selected_rows_copy['年龄111'] = '壮年'
print(selected_rows_copy.head())
# 为了简化代码，这里我们只处理受教育程度为例，你可以参考这个代码处理其他变量
edu_crosstab = pd.crosstab(selected_rows_copy['年龄111'], selected_rows_copy['性别  '], margins=True, margins_name="Total")
print(edu_crosstab)


df_select = df[(df['年龄段_中年'] == 0) & (df['年龄段_壮年'] == 0) & (df['年龄段_老年'] == 1)]

selected_rows_copy = df_select.copy()
selected_rows_copy['年龄111'] = '老年'
print(selected_rows_copy.head())
# 为了简化代码，这里我们只处理受教育程度为例，你可以参考这个代码处理其他变量
edu_crosstab = pd.crosstab(selected_rows_copy['年龄111'], selected_rows_copy['性别  '], margins=True, margins_name="Total")
print(edu_crosstab)


# 手动输入数据
# 数据结构是：[男性数据], [女性数据]
data = np.array([
    [20825, 24244],  # 非饮酒者
    [24351, 18199],   # 现饮酒者
[2264, 1415],
[681, 378],
])

# 使用chi2_contingency进行卡方检验
chi2, p, _, _ = chi2_contingency(data)

print(f"卡方值为: {chi2:.2f}")
print(f"P值为: {p:.4f}")




'''分出流动原因的子样本'''
df_select = df[(df['本次流动原因_家庭类'] == 0) & (df['本次流动原因_工作类'] == 1)]

selected_rows_copy = df_select.copy()
selected_rows_copy['本次流动原因_'] = '工作'
print(selected_rows_copy.head())
# 为了简化代码，这里我们只处理受教育程度为例，你可以参考这个代码处理其他变量
edu_crosstab = pd.crosstab(selected_rows_copy['本次流动原因_'], selected_rows_copy['性别  '], margins=True, margins_name="Total")
print(edu_crosstab)


# 手动输入数据
# 数据结构是：[男性数据], [女性数据]
data = np.array([
    [12380, 9211],  # 非饮酒者
    [33395, 25233],   # 现饮酒者
[2346, 9792],
])

# 使用chi2_contingency进行卡方检验
chi2, p, _, _ = chi2_contingency(data)

print(f"卡方值为: {chi2:.2f}")
print(f"P值为: {p:.4f}")




'''分出歧视的子样本'''
df_select = df[(df['感觉被看不起_同意'] == 1)]

selected_rows_copy = df_select.copy()
selected_rows_copy['歧视'] = '是'
print(selected_rows_copy.head())
# 为了简化代码，这里我们只处理受教育程度为例，你可以参考这个代码处理其他变量
edu_crosstab = pd.crosstab(selected_rows_copy['歧视'], selected_rows_copy['性别  '], margins=True, margins_name="Total")
print(edu_crosstab)

# 手动输入数据
# 数据结构是：[男性数据], [女性数据]
data = np.array([
    [8462, 8223],  # 非饮酒者
    [39659, 36013],   # 现饮酒者
])

# 使用chi2_contingency进行卡方检验
chi2, p, _, _ = chi2_contingency(data)

print(f"卡方值为: {chi2:.2f}")
print(f"P值为: {p:.4f}")


dff = pd.read_csv('跑模型之前的整理（尚未添加虚拟变量）.csv',encoding='GBK')
print(max(dff['年龄'].unique()),min(dff['年龄'].unique()))
print(np.mean(dff['年龄']))


'''流入城市层面变量描述性统计'''

df = pd.read_csv('logit模型数据表（不包含性别交叉项)+城市层面.csv',
                 usecols=['distance','所在城市常住人口（万人）','所在城市GDP（亿元）','所在城市一般公共预算支出（万元）',
                          '所在城市教育支出（万元）','所在城市科技支出（万元）','所在城市二产比重'],
                 encoding='GBK')
print(df.head())
df = df.dropna().reset_index()

desc_stats = df.describe()
desc_stats.to_csv("城市层面变量描述性统计.csv")


















'''卡方检验03.22'''


# 手动输入数据
# 数据结构是：[男性数据], [女性数据]
data = np.array([
    [24841, 23374],  # 非饮酒者
    [24018, 20166],   # 现饮酒者
])

# 使用chi2_contingency进行卡方检验
chi2, p, _, _ = chi2_contingency(data)

print(f"卡方值为: {chi2:.2f}")
print(f"P值为: {p:.4f}")


# 数据结构是：[年龄]
data = np.array([
    [31934, 24965],  # 非饮酒者
    [15036, 16897],   # 现饮酒者
    [1889, 1678],
])

# 使用chi2_contingency进行卡方检验
chi2, p, _, _ = chi2_contingency(data)

print(f"卡方值为: {chi2:.2f}")
print(f"P值为: {p:.4f}")







# 数据结构是：[户口]
data = np.array([
    [37679, 39796],  # 非饮酒者
    [11180, 3744],   # 现饮酒者
])

# 使用chi2_contingency进行卡方检验
chi2, p, _, _ = chi2_contingency(data)

print(f"卡方值为: {chi2:.2f}")
print(f"P值为: {p:.4f}")






# 数据结构是：[婚姻]
data = np.array([
    [7301, 5890],  # 非饮酒者
    [41558, 37650],   # 现饮酒者
])

# 使用chi2_contingency进行卡方检验
chi2, p, _, _ = chi2_contingency(data)

print(f"卡方值为: {chi2:.2f}")
print(f"P值为: {p:.4f}")





# 数据结构是：[教育]
data = np.array([
    [6216, 8639],  # 非饮酒者
    [30695, 30072],   # 现饮酒者
    [11948, 4829],
])

# 使用chi2_contingency进行卡方检验
chi2, p, _, _ = chi2_contingency(data)

print(f"卡方值为: {chi2:.2f}")
print(f"P值为: {p:.4f}")






# 数据结构是：[跨越]
data = np.array([
    [6819, 10115],  # 非饮酒者
    [14856, 12883],   # 现饮酒者
    [27184, 20542],
])

# 使用chi2_contingency进行卡方检验
chi2, p, _, _ = chi2_contingency(data)

print(f"卡方值为: {chi2:.2f}")
print(f"P值为: {p:.4f}")



# 数据结构是：[包容]
data = np.array([
    [8100, 8493],  # 非饮酒者
    [40759, 35047],   # 现饮酒者
])

# 使用chi2_contingency进行卡方检验
chi2, p, _, _ = chi2_contingency(data)

print(f"卡方值为: {chi2:.2f}")
print(f"P值为: {p:.4f}")
