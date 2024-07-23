import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd

# 请求网页内容
url = 'https://www.mca.gov.cn/mzsj/xzqh/1980/201903/201903011447.html'
response = requests.get(url)
response.encoding = 'utf-8'  # 设置编码，以正确解析中文

# 使用BeautifulSoup解析网页
soup = BeautifulSoup(response.text, 'html.parser')

# 找到目标表格
table = soup.find('table')

# 创建CSV文件
with open('行政区划代码.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['行政区划代码', '单位名称'])  # 写入表头

    # 遍历表格的每一行
    for row in table.find_all('tr'):
        cells = row.find_all(['td', 'th'])
        if len(cells) == 2:
            code = cells[0].get_text().strip()  # 获取行政区划代码
            name = cells[1].get_text().strip()  # 获取单位名称
            csv_writer.writerow([code, name])  # 写入CSV文件

print("爬取完成，数据已保存为'行政区划代码.csv'")




