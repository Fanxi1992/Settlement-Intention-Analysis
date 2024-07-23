import pandas as pd
import numpy as np



# 读取CSV文件
file_path = '全样本GWLR_regression_results_加上城市层面变量之后0315_去掉一个奇异值.csv'
df = pd.read_csv(file_path,encoding='GBK')

# 显示数据的前几行，以便了解数据结构和列名称
print(df.head())
print(len(df))


# P值列和对应的参数列
p_columns = ['P值_截距项','P值_性别_女', 'P值_年龄段_中年', 'P值_年龄段_青年','P值_受教育程度_中学学历','P值_受教育程度_大学及以上学历','P值_户口性质_农业','P值_婚姻状况_已婚',
          'P值_感觉被看不起_同意','P值_跨越尺度_省内跨市','P值_跨越尺度_跨省','P值_distance','P值_所在城市常住人口（万人）','P值_所在城市GDP_亿元',
          'P值_所在城市在岗职工平均工资_元','P值_所在城市第三产业比重','P值_所在城市PM2.5','P值_流动时长','P值_同住的家庭成员人数         ']
param_columns = ['参数_截距项','参数_性别_女', '参数_年龄段_中年', '参数_年龄段_青年','参数_受教育程度_中学学历','参数_受教育程度_大学及以上学历','参数_户口性质_农业','参数_婚姻状况_已婚',
          '参数_感觉被看不起_同意','参数_跨越尺度_省内跨市','参数_跨越尺度_跨省','参数_distance','参数_所在城市常住人口（万人）','参数_所在城市GDP_亿元',
          '参数_所在城市在岗职工平均工资_元','参数_所在城市第三产业比重','参数_所在城市PM2.5','参数_流动时长','参数_同住的家庭成员人数         ']

# 循环处理每一组P值和参数列
for p_col, param_col in zip(p_columns, param_columns):
    # 复制df以避免直接修改原始数据
    temp_df = df[['城市位置', p_col, param_col]].copy()
    # 对P值不小于0.1的行，将对应的参数列值改为"Not significant"
    temp_df.loc[temp_df[p_col] >= 0.1, param_col] = "Not significant"
    # 保存到CSV文件，这里只是显示如何命名文件，实际保存代码需要在您的本地环境执行
    file_name = f'{p_col}_处理后.csv'
    print(f"保存文件名: {file_name}")
    temp_df.to_csv(file_name, index=False) # 实际执行时取消注释



# 处理城市位置变量两列经纬度
import pandas as pd
import glob
import os

# 指定源文件夹和目标文件夹路径
source_folder = '待处理参数数据'
target_folder = '处理后参数数据'

# 使用glob模块找到源文件夹下的所有CSV文件
csv_files = glob.glob(os.path.join(source_folder, '*.csv'))

# 遍历所有找到的CSV文件
for file_path in csv_files:
    # 读取CSV文件
    gwr_model = pd.read_csv(file_path)

    # 执行之前的数据处理步骤
    gwr_model['城市位置'] = gwr_model['城市位置'].str.replace('[', '').str.replace(']', '')
    gwr_model[['当地经度', '当地纬度']] = gwr_model['城市位置'].str.split(r'\s+', expand=True).iloc[:, :2].astype(float)

    # 构建目标文件路径
    base_name = os.path.basename(file_path)  # 获取文件的基本名字
    target_file_path = os.path.join(target_folder, base_name)  # 目标文件完整路径

    # 保存处理后的DataFrame到目标文件夹
    gwr_model.to_csv(target_file_path, index=False)






# 合并城市的名字，根据经纬度
# 指定源文件夹路径（包含要合并的基础模型文件）和目标文件夹路径
source_folder = '处理后参数数据'
target_folder = '处理后参数数据_合并了城市名'
city_data_path = 'logit模型数据表（不包含性别交叉项)+城市层面+去空值并标准化0314.csv'

# 读取城市数据
city_data = pd.read_csv(city_data_path, usecols=['出发市', '当地经度']).drop_duplicates(subset=['当地经度'])

# 使用glob找到所有要处理的文件
files = glob.glob(os.path.join(source_folder, '*.csv'))

# 确保目标文件夹存在
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 对每个文件执行合并操作
for file_path in files:
    # 读取基础模型数据
    base_model = pd.read_csv(file_path)

    # 执行合并操作
    merged_data = base_model.merge(city_data, on='当地经度', how='left')

    # 构建目标文件路径
    file_name = os.path.basename(file_path)
    target_file_path = os.path.join(target_folder, file_name)

    # 保存合并后的数据到目标文件夹
    merged_data.to_csv(target_file_path, index=False)

print("所有文件处理完成，并保存到目标文件夹。")








# 将18个文件合并成一个文件，便于绘图
import pandas as pd
import os

# 假设所有CSV文件都在这个文件夹中
folder_path = '处理后参数数据_合并了城市名'

# 获取文件夹中所有CSV文件的路径
csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
print(csv_files,len(csv_files))
# 读取第一个CSV文件，并为后续合并做准备
merged_df = pd.read_csv(csv_files[0],encoding='GBK')

# 循环遍历剩余的CSV文件
for csv_file in csv_files[1:]:
    # 读取CSV文件
    df = pd.read_csv(csv_file,encoding='GBK')
    # 获取独特的参数列名（假设它是除了'当地经度'、'当地纬度'、'出发市'之外的唯一列）
    unique_col = [col for col in df.columns if col not in ['当地经度', '当地纬度', '出发市']][0]
    # 将当前DataFrame与之前合并的DataFrame合并，仅保留独特的参数列
    merged_df = pd.merge(merged_df, df[['当地经度', '当地纬度', '出发市', unique_col]], on=['当地经度', '当地纬度', '出发市'])

# 输出合并后的DataFrame以检查
print(merged_df.head())
print(merged_df)

# 可选：如果你想将合并后的DataFrame保存为一个新的CSV文件
merged_df.to_csv('18张参数空间分布图的合并表.csv', index=False)







