import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import statsmodels.api as sm
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# 高斯核函数（用于计算权重）
def gaussian_kernel(u, v, bandwidth):
    d = np.sqrt(u ** 2 + v ** 2)
    return np.exp(-0.5 * (d / bandwidth) ** 2)

# 计算每个观测点的权重
def calculate_weights(target_point, coordinates, bandwidth):
    # 初始化一个字典用于存储已计算的运算结果
    computed_results = {}
    # 初始化一个空列表用于存储所有的运算结果
    final_results = []

    for value in coordinates:
        value_tuple = tuple(value)
        if value_tuple in computed_results:
            result = computed_results[value_tuple]
        else:
            u = target_point[0] - value_tuple[0]
            v = target_point[1] - value_tuple[1]
            result = gaussian_kernel(u, v, bandwidth)
            computed_results[value_tuple] = result
        final_results.append(result)
    full_weights = np.array(final_results)
    return full_weights


# 修改局部Logistic回归函数以使用预计算的权重
def local_logistic_regression(target_point, X, y, coordinates, bandwidth):
    # 计算权重
    w = calculate_weights(target_point, coordinates, bandwidth)

    # 添加常数项
    X_with_const = sm.add_constant(X)

    # 使用WLS进行拟合
    model = sm.GLM(y, X_with_const, family=sm.families.Binomial(), var_weights=w)
    result = model.fit()

    # 返回参数和似然函数值
    return result.params, result.llf


# AIC准则（用于带宽选择）
def aic(target_point, X, y, coordinates, bandwidth):
    params, llf = local_logistic_regression(target_point, X, y, coordinates, bandwidth)
    return -2 * llf + 2 * len(params)    # 计算AIC


# 全局AIC（用于全局最佳带宽选择）
def global_aic(bandwidth, X, y, coordinates):
    total_aic = 0
    seen = set()  # 用于存储已经遇到的数值
    unique_indices = []  # 用于存储独特数值在原列表中的索引
    # 使用 enumerate 函数遍历列表，这样我们可以同时得到数值和它们的索引
    for index, coord in enumerate(coordinates):
        # 如果这个数值是首次出现
        coord_tuple = tuple(coord)
        if coord_tuple not in seen:
            # 将它加入到已经遇到的数值的集合中
            seen.add(coord_tuple)
            # 存储这个数值在原列表中的索引
            unique_indices.append(index)
    # print(len(unique_indices))
    for i in unique_indices:
        # print(i)
        target_point = coordinates[i]
        total_aic += aic(target_point, X, y, coordinates, bandwidth)
    print(total_aic,bandwidth)
    return total_aic




def local_logistic_regression_final(target_point, X, y, coordinates, bandwidth, x_local, y_local):
    # 计算权重
    w = calculate_weights(target_point, coordinates, bandwidth)

#   X_binary = [1 if x > 0.5 else 0 for x in X]
    #
    # # 第二步：计算 X_binary 和 Y 相同的元素数量
    # count_same = sum([1 for x, y in zip(X_binary, Y) if x == y])

    # 添加常数项
    X_with_const = sm.add_constant(X)
    # print(X.shape,type(X))
    # print(X_with_const.shape)
    # print(x_local.shape,type(x_local))
    x_local_with_const = np.hstack([np.ones((x_local.shape[0], 1)), x_local])
    # print(x_local_with_const.shape)
    # 使用WLS进行拟合
    model = sm.GLM(y, X_with_const, family=sm.families.Binomial(), var_weights=w)
    result = model.fit()
    prediction_prob = result.predict(x_local_with_const)
    X_binary = [1 if x > 0.5 else 0 for x in prediction_prob]
    count_same = sum([1 for x, y in zip(X_binary, y_local) if x == y])
    print(len(prediction_prob),len(y_local))
    print(count_same)
    # prediction_category = 1 if prediction_prob > 0.5 else 0  # 使用0.5作为阈值
    # 返回参数和似然函数值
    return result, count_same



def overall_accuracy(Y_actual, Y_predicted):
    correct_count = np.sum(Y_actual == Y_predicted)
    total_count = len(Y_actual)
    return correct_count / total_count



# 加载数据
file_path = 'logit模型数据表（不包含性别交叉项)+城市层面+去空值并标准化0314.csv'
df = pd.read_csv(file_path)
print(df.head())
print(len(df['现居住地址市（地区）          '].unique()))

#precomputed_weights = {}
# 指定响应变量和自变量
response_var = '如果您符合本地落户条件，您是否愿意把户口迁入本地                        '
predictor_vars = ['性别_女', '年龄段_中年', '年龄段_青年','受教育程度_中学学历','受教育程度_大学及以上学历','户口性质_农业','婚姻状况_已婚',
          '感觉被看不起_同意','跨越尺度_省内跨市','跨越尺度_跨省','distance','所在城市常住人口（万人）','所在城市GDP_亿元',
          '所在城市在岗职工平均工资_元','所在城市第三产业比重','所在城市PM2.5','流动时长','同住的家庭成员人数         ']

# 地理坐标
longitude_var = '当地经度'
latitude_var = '当地纬度'

# 提取变量
Y = df[response_var].values
X = df[predictor_vars].values
coordinates = df[[longitude_var, latitude_var]].values
#
# # 对distance进行标准化
# scaler = StandardScaler()
# X[:, -1] = scaler.fit_transform(X[:, -1].reshape(-1, 1)).flatten()
# print(X[:, -1])

# 通过全局AIC优化带宽
result = minimize_scalar(lambda bw: global_aic(bw, X, Y, coordinates),
                         bounds=(0.1, 10000),
                         method='bounded')

# 最佳全局带宽
best_global_bandwidth = result.x
print(best_global_bandwidth)
print('寻找最佳带宽结束')
print(best_global_bandwidth)

# best_global_bandwidth = 3.777192751
best_global_bandwidth = 2.2931784
# 使用最佳全局带宽进行局部Logistic回归
final_result = []
final_predictions = 0 # 用于存储每个观察点的预测值
seen = set()  # 用于存储已经遇到的数值
unique_indices = []  # 用于存储独特数值在原列表中的索引
# 使用 enumerate 函数遍历列表，这样我们可以同时得到数值和它们的索引
for index, coord in enumerate(coordinates):
    # 如果这个数值是首次出现
    coord_tuple = tuple(coord)
    if coord_tuple not in seen:
        # 将它加入到已经遇到的数值的集合中
        seen.add(coord_tuple)
        # 存储这个数值在原列表中的索引
        unique_indices.append(index)
# print(len(unique_indices))
for i in unique_indices:
    # print(i)
    target_point = coordinates[i]
    # 在这个位置把对应的X和Y全部挑出来，然后输入模型
    select_df = df[df['当地经度'] == df['当地经度'][i]]
    X_local = select_df[predictor_vars].values
    Y_local = select_df[response_var].values
    # print((X_local == 1).all(axis=0))
    local_result, count = local_logistic_regression_final(target_point, X, Y, coordinates, best_global_bandwidth, X_local, Y_local)
    final_result.append(local_result)
#   对所有的count求和，然后和总数进行比例
    final_predictions = final_predictions + count


accuracy = final_predictions/len(df)
# 计算整体准确率
# accuracy = overall_accuracy(Y, final_predictions)
print(f'整体准确率为：{accuracy * 100:.2f}%')

# 定义自变量的列表
predictor_vars = ['性别_女', '年龄段_中年', '年龄段_青年','受教育程度_中学学历','受教育程度_大学及以上学历','户口性质_农业','婚姻状况_已婚',
          '感觉被看不起_同意','跨越尺度_省内跨市','跨越尺度_跨省','distance','所在城市常住人口（万人）','所在城市GDP_亿元',
          '所在城市在岗职工平均工资_元','所在城市第三产业比重','所在城市PM2.5','流动时长','同住的家庭成员人数         ']

# 创建一个空的列表来存储输出数据
output_data = []

# 添加CSV文件的标题行
header = ['城市位置'] + ['参数_' + var for var in ['截距项'] + predictor_vars] + ['P值_' + var for var in ['截距项'] + predictor_vars]
output_data.append(header)

# 循环遍历每个结果，并收集必要的数据
for i, result in enumerate(final_result):
    row = []
    # 添加城市位置
    row.append(coordinates[unique_indices[i]])
    # 添加参数值
    row.extend(result.params)
    # 添加对应的P值
    row.extend(result.pvalues)
    # 将这一行数据添加到output_data中
    output_data.append(row)

# 写入CSV文件
with open('全样本GWLR_regression_results_加上城市层面变量之后0315.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(output_data)










'''模型1：只包含个体层面变量'''
# 加载数据
file_path = 'logit模型数据表（不包含性别交叉项)+城市层面+去空值并标准化0314.csv'
df = pd.read_csv(file_path)
print(df.head())
print(len(df['现居住地址市（地区）          '].unique()))

#precomputed_weights = {}
# 指定响应变量和自变量
response_var = '如果您符合本地落户条件，您是否愿意把户口迁入本地                        '
predictor_vars = ['性别_女', '年龄段_中年', '年龄段_青年','受教育程度_中学学历','受教育程度_大学及以上学历','户口性质_农业','婚姻状况_已婚']

# 地理坐标
longitude_var = '当地经度'
latitude_var = '当地纬度'

# 提取变量
Y = df[response_var].values
X = df[predictor_vars].values
coordinates = df[[longitude_var, latitude_var]].values
#
# # 对distance进行标准化
# scaler = StandardScaler()
# X[:, -1] = scaler.fit_transform(X[:, -1].reshape(-1, 1)).flatten()
# print(X[:, -1])

# 通过全局AIC优化带宽
result = minimize_scalar(lambda bw: global_aic(bw, X, Y, coordinates),
                         bounds=(0.1, 10000),
                         method='bounded')

# 最佳全局带宽
best_global_bandwidth = result.x
print(best_global_bandwidth)
print('寻找最佳带宽结束')
print(best_global_bandwidth)

# best_global_bandwidth = 3.777192751
best_global_bandwidth = 0.31624224
# 使用最佳全局带宽进行局部Logistic回归
final_result = []
final_predictions = 0 # 用于存储每个观察点的预测值
seen = set()  # 用于存储已经遇到的数值
unique_indices = []  # 用于存储独特数值在原列表中的索引
# 使用 enumerate 函数遍历列表，这样我们可以同时得到数值和它们的索引
for index, coord in enumerate(coordinates):
    # 如果这个数值是首次出现
    coord_tuple = tuple(coord)
    if coord_tuple not in seen:
        # 将它加入到已经遇到的数值的集合中
        seen.add(coord_tuple)
        # 存储这个数值在原列表中的索引
        unique_indices.append(index)
# print(len(unique_indices))
for i in unique_indices:
    # print(i)
    target_point = coordinates[i]
    # 在这个位置把对应的X和Y全部挑出来，然后输入模型
    select_df = df[df['当地经度'] == df['当地经度'][i]]
    X_local = select_df[predictor_vars].values
    Y_local = select_df[response_var].values
    # print((X_local == 1).all(axis=0))
    local_result, count = local_logistic_regression_final(target_point, X, Y, coordinates, best_global_bandwidth, X_local, Y_local)
    final_result.append(local_result)
#   对所有的count求和，然后和总数进行比例
    final_predictions = final_predictions + count


accuracy = final_predictions/len(df)
# 计算整体准确率
# accuracy = overall_accuracy(Y, final_predictions)
print(f'整体准确率为：{accuracy * 100:.2f}%')





















'''
# ... （前面的代码不变）

# 计算整体的预测准确率
def overall_accuracy(Y_actual, Y_predicted):
    correct_count = np.sum(Y_actual == Y_predicted)
    total_count = len(Y_actual)
    return correct_count / total_count

# ... （中间的代码不变）

# 使用最佳全局带宽进行局部Logistic回归，同时收集每个点的预测值
final_result = []
final_predictions = np.zeros(Y.shape)  # 用于存储每个观察点的预测值

for i in range(len(coordinates)):
    target_point = coordinates[i]
    local_result = local_logistic_regression_final(target_point, X, Y, coordinates, best_global_bandwidth)
    final_result.append(local_result)

    # 使用模型预测
    X_with_const = sm.add_constant(X[i, :])
    prediction_prob = local_result.predict(X_with_const)
    prediction_class = 1 if prediction_prob > 0.5 else 0  # 使用0.5作为阈值
    final_predictions[i] = prediction_class

# 计算整体准确率
accuracy = overall_accuracy(Y, final_predictions)
print(f'整体准确率为：{accuracy * 100:.2f}%')

# ... （后面的代码不变）

'''







'''分析男性样本'''

# 加载数据
file_path = 'logit模型数据表（不包含性别交叉项)+城市层面+去空值并标准化.csv'
df = pd.read_csv(file_path)
print(df.head())
print(len(df['现居住地址市（地区）          '].unique()))


data = df[df['性别_女'] == 0].copy()  # 使用copy()创建df的一个副本
data.reset_index(drop=True, inplace=True)  # 重置索引并删除旧索引列

print(len(data))

df = data
print(len(df))
#precomputed_weights = {}
# 指定响应变量和自变量
response_var = '如果您符合本地落户条件，您是否愿意把户口迁入本地                        '
predictor_vars = ['年龄段_中年', '年龄段_壮年', '年龄段_老年',
                  '受教育程度_中学学历', '受教育程度_大学专科', '受教育程度_本科及以上学历',
                  '户口性质_农业', '婚姻状况_已婚', '婚姻状况_离婚',
                  '本次流动原因_家庭类', '本次流动原因_工作类',
                  '感觉被看不起_同意', 'distance','所在城市常住人口（万人）','所在城市GDP（亿元）',
                '所在城市一般公共预算支出（万元）','所在城市教育支出（万元）','所在城市科技支出（万元）','所在城市二产比重']

# 地理坐标
longitude_var = '当地经度'
latitude_var = '当地纬度'

# 提取变量
Y = df[response_var].values
X = df[predictor_vars].values
coordinates = df[[longitude_var, latitude_var]].values
#
# # 对distance进行标准化
# scaler = StandardScaler()
# X[:, -1] = scaler.fit_transform(X[:, -1].reshape(-1, 1)).flatten()
# print(X[:, -1])

# 通过全局AIC优化带宽
result = minimize_scalar(lambda bw: global_aic(bw, X, Y, coordinates),
                         bounds=(0.1, 10000),
                         method='bounded')

# 最佳全局带宽
best_global_bandwidth = result.x
print(best_global_bandwidth)
print('寻找最佳带宽结束')
print(best_global_bandwidth)

best_global_bandwidth = 3.54347609

# 使用最佳全局带宽进行局部Logistic回归
final_result = []
final_predictions = 0 # 用于存储每个观察点的预测值
seen = set()  # 用于存储已经遇到的数值
unique_indices = []  # 用于存储独特数值在原列表中的索引
# 使用 enumerate 函数遍历列表，这样我们可以同时得到数值和它们的索引
for index, coord in enumerate(coordinates):
    # 如果这个数值是首次出现
    coord_tuple = tuple(coord)
    if coord_tuple not in seen:
        # 将它加入到已经遇到的数值的集合中
        seen.add(coord_tuple)
        # 存储这个数值在原列表中的索引
        unique_indices.append(index)
# print(len(unique_indices))
for i in unique_indices:
    # print(i)
    target_point = coordinates[i]
    # 在这个位置把对应的X和Y全部挑出来，然后输入模型
    select_df = df[df['当地经度'] == df['当地经度'][i]]
    X_local = select_df[predictor_vars].values
    Y_local = select_df[response_var].values
    # print((X_local == 1).all(axis=0))
    local_result, count = local_logistic_regression_final(target_point, X, Y, coordinates, best_global_bandwidth, X_local, Y_local)
    final_result.append(local_result)
#   对所有的count求和，然后和总数进行比例
    final_predictions = final_predictions + count


accuracy = final_predictions/len(df)
# 计算整体准确率
# accuracy = overall_accuracy(Y, final_predictions)
print(f'整体准确率为：{accuracy * 100:.2f}%')

# 定义自变量的列表
predictor_vars = ['年龄段_中年', '年龄段_壮年', '年龄段_老年',
                  '受教育程度_中学学历', '受教育程度_大学专科', '受教育程度_本科及以上学历',
                  '户口性质_农业', '婚姻状况_已婚', '婚姻状况_离婚',
                  '本次流动原因_家庭类', '本次流动原因_工作类',
                  '感觉被看不起_同意', 'distance','所在城市常住人口（万人）','所在城市GDP（亿元）',
                '所在城市一般公共预算支出（万元）','所在城市教育支出（万元）','所在城市科技支出（万元）','所在城市二产比重']

# 创建一个空的列表来存储输出数据
output_data = []

# 添加CSV文件的标题行
header = ['城市位置'] + ['参数_' + var for var in ['截距项'] + predictor_vars] + ['P值_' + var for var in ['截距项'] + predictor_vars]
output_data.append(header)

# 循环遍历每个结果，并收集必要的数据
for i, result in enumerate(final_result):
    row = []
    # 添加城市位置
    row.append(coordinates[unique_indices[i]])
    # 添加参数值
    row.extend(result.params)
    # 添加对应的P值
    row.extend(result.pvalues)
    # 将这一行数据添加到output_data中
    output_data.append(row)

# 写入CSV文件
with open('男性样本GWLR_regression_results_加上城市层面变量之后.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(output_data)




































'''分析女性样本'''

# 加载数据
file_path = 'logit模型数据表（不包含性别交叉项)+城市层面+去空值并标准化.csv'
df = pd.read_csv(file_path)
print(df.head())
print(len(df['现居住地址市（地区）          '].unique()))


data = df[df['性别_女'] == 1].copy()  # 使用copy()创建df的一个副本
data.reset_index(drop=True, inplace=True)  # 重置索引并删除旧索引列

print(len(data))

df = data
print(len(df))
#precomputed_weights = {}
# 指定响应变量和自变量
response_var = '如果您符合本地落户条件，您是否愿意把户口迁入本地                        '
predictor_vars = ['年龄段_中年', '年龄段_壮年', '年龄段_老年',
                  '受教育程度_中学学历', '受教育程度_大学专科', '受教育程度_本科及以上学历',
                  '户口性质_农业', '婚姻状况_已婚', '婚姻状况_离婚',
                  '本次流动原因_家庭类', '本次流动原因_工作类',
                  '感觉被看不起_同意', 'distance','所在城市常住人口（万人）','所在城市GDP（亿元）',
                '所在城市一般公共预算支出（万元）','所在城市教育支出（万元）','所在城市科技支出（万元）','所在城市二产比重']

# 地理坐标
longitude_var = '当地经度'
latitude_var = '当地纬度'

# 提取变量
Y = df[response_var].values
X = df[predictor_vars].values
coordinates = df[[longitude_var, latitude_var]].values
#
# # 对distance进行标准化
# scaler = StandardScaler()
# X[:, -1] = scaler.fit_transform(X[:, -1].reshape(-1, 1)).flatten()
# print(X[:, -1])

# 通过全局AIC优化带宽
result = minimize_scalar(lambda bw: global_aic(bw, X, Y, coordinates),
                         bounds=(0.1, 10000),
                         method='bounded')

# 最佳全局带宽
best_global_bandwidth = result.x
print(best_global_bandwidth)
print('寻找最佳带宽结束')
print(best_global_bandwidth)

best_global_bandwidth = 3.970448

# 使用最佳全局带宽进行局部Logistic回归
final_result = []
final_predictions = 0 # 用于存储每个观察点的预测值
seen = set()  # 用于存储已经遇到的数值
unique_indices = []  # 用于存储独特数值在原列表中的索引
# 使用 enumerate 函数遍历列表，这样我们可以同时得到数值和它们的索引
for index, coord in enumerate(coordinates):
    # 如果这个数值是首次出现
    coord_tuple = tuple(coord)
    if coord_tuple not in seen:
        # 将它加入到已经遇到的数值的集合中
        seen.add(coord_tuple)
        # 存储这个数值在原列表中的索引
        unique_indices.append(index)
# print(len(unique_indices))
for i in unique_indices:
    # print(i)
    target_point = coordinates[i]
    # 在这个位置把对应的X和Y全部挑出来，然后输入模型
    select_df = df[df['当地经度'] == df['当地经度'][i]]
    X_local = select_df[predictor_vars].values
    Y_local = select_df[response_var].values
    # print((X_local == 1).all(axis=0))
    local_result, count = local_logistic_regression_final(target_point, X, Y, coordinates, best_global_bandwidth, X_local, Y_local)
    final_result.append(local_result)
#   对所有的count求和，然后和总数进行比例
    final_predictions = final_predictions + count


accuracy = final_predictions/len(df)
# 计算整体准确率
# accuracy = overall_accuracy(Y, final_predictions)
print(f'整体准确率为：{accuracy * 100:.2f}%')

# 定义自变量的列表
predictor_vars = ['年龄段_中年', '年龄段_壮年', '年龄段_老年',
                  '受教育程度_中学学历', '受教育程度_大学专科', '受教育程度_本科及以上学历',
                  '户口性质_农业', '婚姻状况_已婚', '婚姻状况_离婚',
                  '本次流动原因_家庭类', '本次流动原因_工作类',
                  '感觉被看不起_同意', 'distance','所在城市常住人口（万人）','所在城市GDP（亿元）',
                '所在城市一般公共预算支出（万元）','所在城市教育支出（万元）','所在城市科技支出（万元）','所在城市二产比重']

# 创建一个空的列表来存储输出数据
output_data = []

# 添加CSV文件的标题行
header = ['城市位置'] + ['参数_' + var for var in ['截距项'] + predictor_vars] + ['P值_' + var for var in ['截距项'] + predictor_vars]
output_data.append(header)

# 循环遍历每个结果，并收集必要的数据
for i, result in enumerate(final_result):
    row = []
    # 添加城市位置
    row.append(coordinates[unique_indices[i]])
    # 添加参数值
    row.extend(result.params)
    # 添加对应的P值
    row.extend(result.pvalues)
    # 将这一行数据添加到output_data中
    output_data.append(row)

# 写入CSV文件
with open('女性样本GWLR_regression_results_加上城市层面变量之后.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(output_data)

