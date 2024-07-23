import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import statsmodels.api as sm
import numpy as np
import pysal
from esda.moran import Moran
import libpysal
from libpysal.weights import DistanceBand
import geopandas as gpd
from shapely.geometry import Point

# 描述性统计先
file_path = 'logit模型数据表（不包含性别交叉项)+城市层面0314.csv'
data = pd.read_csv(file_path)
print(data.head())

data = data.dropna().reset_index()
print(len(data))
print(data)

settle_counts = data['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].value_counts()


# 2. 统计性别列中，值为“男”和“女”分别有多少行
gender_counts = data['性别  '].value_counts()

# 3. 统计一下性别为女的数据行中，愿意落户和不愿意落户的行分别有多少
female_settle_counts = data[data['性别  '] == '女']['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].value_counts()
male_settle_counts = data[data['性别  '] == '男']['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].value_counts()
print(male_settle_counts)
# 打印结果
print("是否愿意落户统计：\n", settle_counts)
print("性别统计：\n", gender_counts)
print("性别为女的愿意与不愿意落户统计：\n", female_settle_counts)


# 2. 统计性别列中，值为“男”和“女”分别有多少行
gender_counts = data['年龄段'].value_counts()
# 3. 统计一下性别为女的数据行中，愿意落户和不愿意落户的行分别有多少
female_settle_counts = data[data['年龄段'] == '青年']['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].value_counts()
male_settle_counts = data[data['年龄段'] == '中年']['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].value_counts()
male1_settle_counts = data[data['年龄段'] == '老年']['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].value_counts()
print(gender_counts)
print(female_settle_counts)
print(male_settle_counts)
print(male1_settle_counts)



# 2. 统计性别列中，值为“男”和“女”分别有多少行
gender_counts = data['户口性质    '].value_counts()
# 3. 统计一下性别为女的数据行中，愿意落户和不愿意落户的行分别有多少
female_settle_counts = data[data['户口性质    '] == '农业']['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].value_counts()
male_settle_counts = data[data['户口性质    '] == '非农业']['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].value_counts()
print(gender_counts)
print(female_settle_counts)
print(male_settle_counts)




# 2. 统计性别列中，值为“男”和“女”分别有多少行
gender_counts = data['婚姻状况    '].value_counts()
# 3. 统计一下性别为女的数据行中，愿意落户和不愿意落户的行分别有多少
female_settle_counts = data[data['婚姻状况    '] == '未婚']['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].value_counts()
male_settle_counts = data[data['婚姻状况    '] == '已婚']['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].value_counts()
print(gender_counts)
print(female_settle_counts)
print(male_settle_counts)






# 2. 统计性别列中，值为“男”和“女”分别有多少行
gender_counts = data['受教育程度     '].value_counts()
# 3. 统计一下性别为女的数据行中，愿意落户和不愿意落户的行分别有多少
female_settle_counts = data[data['受教育程度     '] == '小学及以下学历']['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].value_counts()
male_settle_counts = data[data['受教育程度     '] == '中学学历']['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].value_counts()
male1_settle_counts = data[data['受教育程度     '] == '大学及以上学历']['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].value_counts()

print(gender_counts)
print(female_settle_counts)
print(male_settle_counts)
print(male1_settle_counts)







# 2. 统计性别列中，值为“男”和“女”分别有多少行
gender_counts = data['本次流动范围      '].value_counts()
# 3. 统计一下性别为女的数据行中，愿意落户和不愿意落户的行分别有多少
female_settle_counts = data[data['本次流动范围      '] == '市内跨县']['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].value_counts()
male_settle_counts = data[data['本次流动范围      '] == '省内跨市']['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].value_counts()
male1_settle_counts = data[data['本次流动范围      '] == '跨省']['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].value_counts()

print(gender_counts)
print(female_settle_counts)
print(male_settle_counts)
print(male1_settle_counts)




# 2. 统计性别列中，值为“男”和“女”分别有多少行
gender_counts = data['您是否同意“我感觉本地人看不起外地人”这个说法                         '].value_counts()
# 3. 统计一下性别为女的数据行中，愿意落户和不愿意落户的行分别有多少
female_settle_counts = data[data['您是否同意“我感觉本地人看不起外地人”这个说法                         '] == '同意']['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].value_counts()
male_settle_counts = data[data['您是否同意“我感觉本地人看不起外地人”这个说法                         '] == '不同意']['如果您符合本地落户条件，您是否愿意把户口迁入本地                        '].value_counts()
print(gender_counts)
print(female_settle_counts)
print(male_settle_counts)





# 连续变量描述性统计
stats = data[['流动时长', '同住的家庭成员人数         ', '本次流动范围      ','distance','所在城市常住人口（万人）','所在城市GDP_亿元','所在城市在岗职工平均工资_元','所在城市第三产业比重','所在城市PM2.5']].describe().loc[['count', 'min', 'max', 'mean', 'std']]
print(stats)


















# 创建一个示例DataFrame
file_path = 'logit模型数据表（不包含性别交叉项)+城市层面0314.csv'
data = pd.read_csv(file_path)

data = data.dropna().reset_index()



# 初始化标准化器
scaler = StandardScaler()
scaler_column = ['distance','所在城市常住人口（万人）','所在城市GDP_亿元','流动时长','同住的家庭成员人数         ',
                 '所在城市在岗职工平均工资_元','所在城市第三产业比重','所在城市PM2.5']
# 仅对连续变量列进行标准化
data[scaler_column] = scaler.fit_transform(data[scaler_column])

# 定义自变量和因变量
X = data[['性别_女', '年龄段_中年', '年龄段_青年','受教育程度_中学学历','受教育程度_大学及以上学历','户口性质_农业','婚姻状况_已婚',
          '感觉被看不起_同意','跨越尺度_省内跨市','跨越尺度_跨省','distance','所在城市常住人口（万人）','所在城市GDP_亿元',
          '所在城市在岗职工平均工资_元','所在城市第三产业比重','所在城市PM2.5','流动时长','同住的家庭成员人数         ']]
y = data['如果您符合本地落户条件，您是否愿意把户口迁入本地                        ']

# 地理坐标
longitude_var = '当地经度'
latitude_var = '当地纬度'

# 首先，将普通的 DataFrame 转换为 GeoDataFrame
geometry = [Point(xy) for xy in zip(data[longitude_var], data[latitude_var])]
geo_data = gpd.GeoDataFrame(data, geometry=geometry)


# 添加截距项
X = sm.add_constant(X)

# 运行Logit模型
model = sm.Logit(y, X)
result = model.fit()

# 预测
y_pred_proba = result.predict(X)

# 通过设置阈值，将预测概率转换为类别标签
threshold = 0.5
y_pred_label = (y_pred_proba >= threshold).astype(int)

# 计算指标
accuracy = accuracy_score(y, y_pred_label)

# 计算伪R方
pseudo_r_squared = result.prsquared

# 计算AICc
n = len(y)
k = len(result.params)
aic = result.aic
aicc = aic + 2*k*(k+1)/(n-k-1)

print(f'准确率: {accuracy}')  # 准确率
print(f'伪R方: {pseudo_r_squared}')  # 伪R方
print(f'AICc: {aicc}')  # AICc


# 计算模型的残差
residuals = y - y_pred_proba

# 然后使用 GeoDataFrame 创建空间权重矩阵
w = DistanceBand.from_dataframe(geo_data, threshold=1.2)

# 然后继续你的Moran's I计算
moran = Moran(residuals, w)
print(f"Moran's I: {moran.I}")
p_value = moran.p_sim
print(f"Moran's I p-value: {p_value}")


# 查看模型结果
print(result.summary())

params = result.params
llf = result.llf

AIC = -2 * llf + 2 * len(params)
print(AIC)
# 获取Odds Ratio (OR)
print("\nOdds Ratios:")
print(np.exp(result.params))

# 预测
predictions = result.predict(X)
print("\nPredictions:")
print(predictions)











'''对总样本和男女样本分别进行普通logit回归'''

data = pd.read_csv('logit模型数据表（不包含性别交叉项)+城市层面+去空值并标准化0314.csv')
print(data.head())

# 定义自变量和因变量
X = data[['性别_女', '年龄段_中年', '年龄段_青年','受教育程度_中学学历','受教育程度_大学及以上学历','户口性质_农业','婚姻状况_已婚',
          '感觉被看不起_同意','跨越尺度_省内跨市','跨越尺度_跨省','distance','所在城市常住人口（万人）','所在城市GDP_亿元',
          '所在城市在岗职工平均工资_元','所在城市第三产业比重','所在城市PM2.5','流动时长','同住的家庭成员人数         ']]
y = data['如果您符合本地落户条件，您是否愿意把户口迁入本地                        ']

# 添加截距项
X = sm.add_constant(X)

# 运行Logit模型
model = sm.Logit(y, X)
result = model.fit()

# 预测
y_pred_proba = result.predict(X)

# 通过设置阈值，将预测概率转换为类别标签
threshold = 0.5
y_pred_label = (y_pred_proba >= threshold).astype(int)

# 计算指标
accuracy = accuracy_score(y, y_pred_label)
precision = precision_score(y, y_pred_label)
recall = recall_score(y, y_pred_label)
f1 = f1_score(y, y_pred_label)
roc_auc = roc_auc_score(y, y_pred_proba)

print(f'准确率: {accuracy}')  # 准确率
print(f'精确度: {precision}')  # 精确度
print(f'召回率: {recall}')  # 召回率
print(f'F1分数: {f1}')  # F1分数
print(f'ROC AUC: {roc_auc}')  # ROC AUC

# 查看模型结果
print(result.summary())

params = result.params

# 获取Odds Ratio (OR)
print("\nOdds Ratios:")
print(np.exp(result.params))




















'''残差空间自相关性计算'''
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
from esda.moran import Moran
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
import pysal
from esda.moran import Moran
import libpysal
from libpysal.weights import DistanceBand
import geopandas as gpd
from shapely.geometry import Point

# 加载数据

data = pd.read_csv('logit模型数据表（不包含性别交叉项)+城市层面+去空值并标准化0314.csv')
print(data.head())

# 定义自变量和因变量
X = data[['性别_女', '年龄段_中年', '年龄段_青年','受教育程度_中学学历','受教育程度_大学及以上学历','户口性质_农业','婚姻状况_已婚',
          '感觉被看不起_同意','跨越尺度_省内跨市','跨越尺度_跨省','distance','所在城市常住人口（万人）','所在城市GDP_亿元',
          '所在城市在岗职工平均工资_元','所在城市第三产业比重','所在城市PM2.5','流动时长','同住的家庭成员人数         ']]
y = data['如果您符合本地落户条件，您是否愿意把户口迁入本地                        ']

# 添加截距项
X = sm.add_constant(X)

# 运行Logit模型
model = sm.Logit(y, X)
result = model.fit()

# 预测
y_pred_proba = result.predict(X)

# 通过设置阈值，将预测概率转换为类别标签
threshold = 0.5
y_pred_label = (y_pred_proba >= threshold).astype(int)

# 计算模型的残差
residuals = y - y_pred_proba

# 将残差加入原始DataFrame
data['residuals'] = residuals

# 计算每个城市位置的平均残差
average_residuals_by_city = data.groupby(['当地经度', '当地纬度'])['residuals'].mean().reset_index()

# 创建平均残差的GeoDataFrame
geometry = [Point(xy) for xy in zip(average_residuals_by_city['当地经度'], average_residuals_by_city['当地纬度'])]
geo_data_avg = gpd.GeoDataFrame(average_residuals_by_city, geometry=geometry)

# 创建空间权重矩阵
w = DistanceBand.from_dataframe(geo_data_avg, threshold=1.2, binary=False, silence_warnings=True)

# 计算Moran's I
moran = Moran(geo_data_avg['residuals'], w)
print(f"Moran's I: {moran.I}")
print(f"Moran's I p-value: {moran.p_sim}")




























































''' model1'''


data = pd.read_csv('logit模型数据表（不包含性别交叉项)+城市层面+去空值并标准化0314.csv')
print(data.head())

# 定义自变量和因变量
X = data[['性别_女', '年龄段_中年', '年龄段_青年','受教育程度_中学学历','受教育程度_大学及以上学历','户口性质_农业','婚姻状况_已婚']]
y = data['如果您符合本地落户条件，您是否愿意把户口迁入本地                        ']

# 添加截距项
X = sm.add_constant(X)

# 运行Logit模型
model = sm.Logit(y, X)
result = model.fit()

# 预测
y_pred_proba = result.predict(X)

# 通过设置阈值，将预测概率转换为类别标签
threshold = 0.5
y_pred_label = (y_pred_proba >= threshold).astype(int)

# 计算指标
accuracy = accuracy_score(y, y_pred_label)
precision = precision_score(y, y_pred_label)
recall = recall_score(y, y_pred_label)
f1 = f1_score(y, y_pred_label)
roc_auc = roc_auc_score(y, y_pred_proba)

print(f'准确率: {accuracy}')  # 准确率
print(f'精确度: {precision}')  # 精确度
print(f'召回率: {recall}')  # 召回率
print(f'F1分数: {f1}')  # F1分数
print(f'ROC AUC: {roc_auc}')  # ROC AUC

# 查看模型结果
print(result.summary())

params = result.params

# 获取Odds Ratio (OR)
print("\nOdds Ratios:")
print(np.exp(result.params))
















''' model2'''

data = pd.read_csv('logit模型数据表（不包含性别交叉项)+城市层面+去空值并标准化0314.csv')
print(data.head())

# 定义自变量和因变量
X = data[['性别_女', '年龄段_中年', '年龄段_青年','受教育程度_中学学历','受教育程度_大学及以上学历','户口性质_农业','婚姻状况_已婚',
          '跨越尺度_省内跨市','跨越尺度_跨省','distance','流动时长','同住的家庭成员人数         ']]
y = data['如果您符合本地落户条件，您是否愿意把户口迁入本地                        ']

# 添加截距项
X = sm.add_constant(X)

# 运行Logit模型
model = sm.Logit(y, X)
result = model.fit()

# 预测
y_pred_proba = result.predict(X)

# 通过设置阈值，将预测概率转换为类别标签
threshold = 0.5
y_pred_label = (y_pred_proba >= threshold).astype(int)

# 计算指标
accuracy = accuracy_score(y, y_pred_label)
precision = precision_score(y, y_pred_label)
recall = recall_score(y, y_pred_label)
f1 = f1_score(y, y_pred_label)
roc_auc = roc_auc_score(y, y_pred_proba)

print(f'准确率: {accuracy}')  # 准确率
print(f'精确度: {precision}')  # 精确度
print(f'召回率: {recall}')  # 召回率
print(f'F1分数: {f1}')  # F1分数
print(f'ROC AUC: {roc_auc}')  # ROC AUC

# 查看模型结果
print(result.summary())

params = result.params

# 获取Odds Ratio (OR)
print("\nOdds Ratios:")
print(np.exp(result.params))

























'''男性样本'''
df = pd.read_csv('logit模型数据表（不包含性别交叉项)+城市层面+去空值并标准化.csv')
print(df.head())
print(len(df))

data = df[df['性别_女'] == 0]
print(len(data))
# 定义自变量和因变量
X = data[['年龄段_中年', '年龄段_壮年','年龄段_老年','受教育程度_中学学历','受教育程度_大学专科',
          '受教育程度_本科及以上学历','户口性质_农业','婚姻状况_已婚','婚姻状况_离婚','本次流动原因_家庭类',
          '本次流动原因_工作类','感觉被看不起_同意','distance','所在城市常住人口（万人）','所在城市GDP（亿元）',
          '所在城市一般公共预算支出（万元）','所在城市教育支出（万元）','所在城市科技支出（万元）','所在城市二产比重']]
y = data['如果您符合本地落户条件，您是否愿意把户口迁入本地                        ']

# 添加截距项
X = sm.add_constant(X)

# 运行Logit模型
model = sm.Logit(y, X)
result = model.fit()

# 预测
y_pred_proba = result.predict(X)

# 通过设置阈值，将预测概率转换为类别标签
threshold = 0.5
y_pred_label = (y_pred_proba >= threshold).astype(int)

# 计算指标
accuracy = accuracy_score(y, y_pred_label)
precision = precision_score(y, y_pred_label)
recall = recall_score(y, y_pred_label)
f1 = f1_score(y, y_pred_label)
roc_auc = roc_auc_score(y, y_pred_proba)

print(f'准确率: {accuracy}')  # 准确率
print(f'精确度: {precision}')  # 精确度
print(f'召回率: {recall}')  # 召回率
print(f'F1分数: {f1}')  # F1分数
print(f'ROC AUC: {roc_auc}')  # ROC AUC

# 查看模型结果
print(result.summary())

params = result.params

# 获取Odds Ratio (OR)
print("\nOdds Ratios:")
print(np.exp(result.params))





'''女性样本'''

df = pd.read_csv('logit模型数据表（不包含性别交叉项)+城市层面+去空值并标准化.csv')
print(df.head())
print(len(df))

data = df[df['性别_女'] == 1]
print(len(data))
# 定义自变量和因变量
X = data[['年龄段_中年', '年龄段_壮年','年龄段_老年','受教育程度_中学学历','受教育程度_大学专科',
          '受教育程度_本科及以上学历','户口性质_农业','婚姻状况_已婚','婚姻状况_离婚','本次流动原因_家庭类',
          '本次流动原因_工作类','感觉被看不起_同意','distance','所在城市常住人口（万人）','所在城市GDP（亿元）',
          '所在城市一般公共预算支出（万元）','所在城市教育支出（万元）','所在城市科技支出（万元）','所在城市二产比重']]
y = data['如果您符合本地落户条件，您是否愿意把户口迁入本地                        ']

# 添加截距项
X = sm.add_constant(X)

# 运行Logit模型
model = sm.Logit(y, X)
result = model.fit()

# 预测
y_pred_proba = result.predict(X)

# 通过设置阈值，将预测概率转换为类别标签
threshold = 0.5
y_pred_label = (y_pred_proba >= threshold).astype(int)

# 计算指标
accuracy = accuracy_score(y, y_pred_label)
precision = precision_score(y, y_pred_label)
recall = recall_score(y, y_pred_label)
f1 = f1_score(y, y_pred_label)
roc_auc = roc_auc_score(y, y_pred_proba)

print(f'准确率: {accuracy}')  # 准确率
print(f'精确度: {precision}')  # 精确度
print(f'召回率: {recall}')  # 召回率
print(f'F1分数: {f1}')  # F1分数
print(f'ROC AUC: {roc_auc}')  # ROC AUC

# 查看模型结果
print(result.summary())

params = result.params

# 获取Odds Ratio (OR)
print("\nOdds Ratios:")
print(np.exp(result.params))