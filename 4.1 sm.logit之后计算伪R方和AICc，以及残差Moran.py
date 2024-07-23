import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import statsmodels.api as sm
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pysal
from esda.moran import Moran
import libpysal
from libpysal.weights import DistanceBand
import geopandas as gpd
from shapely.geometry import Point

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
    # 计算伪R方和AICc

    # 计算伪R方
    null_model = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), var_weights=w).fit()
    pseudo_r_squared = 1 - result.llf / null_model.llf

    # 计算AICc
    aic = result.aic
    n = len(X)
    k = len(result.params)
    aicc = aic + 2*k*(k+1)/(n-k-1)
    print(aicc)
    print(pseudo_r_squared)

    return result, count_same, pseudo_r_squared, aicc



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

# 初始化伪R方和AICc的列表
pseudo_r_squared_list = []
aicc_list = []


for i in unique_indices:
    target_point = coordinates[i]
    select_df = df[df['当地经度'] == df['当地经度'][i]]
    X_local = select_df[predictor_vars].values
    Y_local = select_df[response_var].values
    local_result, count, pseudo_r_squared, aicc = local_logistic_regression_final(target_point, X, Y, coordinates, best_global_bandwidth, X_local, Y_local)
    final_result.append(local_result)
    final_predictions = final_predictions + count

    # 将伪R方和AICc添加到列表中
    pseudo_r_squared_list.append(pseudo_r_squared)
    aicc_list.append(aicc)

# 计算整个GWLR模型的伪R方和AICc
average_pseudo_r_squared = np.mean(pseudo_r_squared_list)
average_aicc = np.mean(aicc_list)

print(pseudo_r_squared_list)
print(aicc_list)
average_pseudo_r_squared = np.nanmean(pseudo_r_squared_list)
average_aicc = np.nanmean(aicc_list)
print(f'整个GWLR模型的平均伪R方为：{average_pseudo_r_squared}')
print(f'整个GWLR模型的平均AICc为：{average_aicc}')



accuracy = final_predictions/len(df)
# 计算整体准确率
# accuracy = overall_accuracy(Y, final_predictions)
print(f'整体准确率为：{accuracy * 100:.2f}%')











'''计算残差Moran指数'''
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.optimize import minimize_scalar
import statsmodels.api as sm
from libpysal.weights import DistanceBand
from esda.moran import Moran

# 保留之前的高斯核函数和权重计算函数

def gaussian_kernel(u, v, bandwidth):
    d = np.sqrt(u**2 + v**2)
    return np.exp(-0.5 * (d / bandwidth)**2)

def calculate_weights(target_point, coordinates, bandwidth):
    computed_results = {}
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
    return np.array(final_results)

# GWLR模型预测和残差计算
def local_logistic_regression_final(target_point, X, y, coordinates, bandwidth, x_local, y_local):
    w = calculate_weights(target_point, coordinates, bandwidth)
    X_with_const = sm.add_constant(X)
    x_local_with_const = np.hstack([np.ones((x_local.shape[0], 1)), x_local])
    model = sm.GLM(y, X_with_const, family=sm.families.Binomial(), var_weights=w)
    result = model.fit()
    prediction_prob = result.predict(x_local_with_const)
    return result, prediction_prob

# 计算GWLR模型并获取残差
def compute_gwlr_and_residuals(df, predictor_vars, response_var, coordinates, bandwidth):
    residuals = np.zeros(len(df))
    for index, coord in enumerate(coordinates):
        target_point = coord
        X_local = df[predictor_vars].values
        Y_local = df[response_var].values
        local_result, prediction_prob = local_logistic_regression_final(target_point, X_local, Y_local, coordinates, bandwidth, X_local, Y_local)
        residuals[index] = Y_local[index] - prediction_prob[index]
    return residuals

# 计算空间自相关性Moran's I
def calculate_morans_i(df, residuals, longitude_var, latitude_var):
    df['residuals'] = residuals
    geometry = [Point(xy) for xy in zip(df[longitude_var], df[latitude_var])]
    geo_df = gpd.GeoDataFrame(df, geometry=geometry)
    w = DistanceBand.from_dataframe(geo_df, threshold=1.2, binary=False, silence_warnings=True)
    moran = Moran(geo_df['residuals'], w)
    return moran.I, moran.p_sim

# 加载数据和模型参数设置
df = pd.read_csv('logit模型数据表（不包含性别交叉项)+城市层面+去空值并标准化0314.csv')
predictor_vars = [...]  # 自变量列表
response_var = '如果您符合本地落户条件，您是否愿意把户口迁入本地'
coordinates = df[['当地经度', '当地纬度']].values
best_global_bandwidth = 2.2931784  # 之前计算得到的最佳带宽

# 计算残差
residuals = compute_gwlr_and_residuals(df, predictor_vars, response_var, coordinates, best_global_bandwidth)

# 计算Moran's I
moran_I, moran_p_value = calculate_morans_i(df, residuals, '当地经度', '当地纬度')

print(f"Moran's I: {moran_I}")
print(f"Moran's I p-value: {moran_p_value}")












# 假设df已经包含了每个个体的预测残差
# 计算每个城市的平均残差
city_avg_residuals = df.groupby(['当地经度', '当地纬度'])['residuals'].mean().reset_index()

# 创建GeoDataFrame
geometry = [Point(xy) for xy in zip(city_avg_residuals['当地经度'], city_avg_residuals['当地纬度'])]
geo_df = gpd.GeoDataFrame(city_avg_residuals, geometry=geometry)

# 创建空间权重矩阵
w = DistanceBand.from_dataframe(geo_df, threshold=1.2, binary=False, silence_warnings=True)

# 计算Moran's I
moran = Moran(geo_df['residuals'], w)
print(f"Moran's I: {moran.I}")
print(f"Moran's I p-value: {moran.p_sim}")