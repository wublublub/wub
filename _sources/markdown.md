# NBA爬虫数据分析

本次实训期间我个人完成的一个对于网页爬取功能的实现以及在此基础上对于多文件整合的数据分析

## 爬取数据展示(部分)
```python
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('../球队赛程/lakers_schedule.csv')

# 显示数据框
print(df)
```

## 湖人对战掘金历史五个赛季平均胜率

```python
# -*- coding: utf-8 -*-
import pandas as pd

files = ['2023','2022','2021','2020','2019']

for file in files:
    # 假设CSV文件已经保存在这个路径
    file_path = f'./{file}.csv'

    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 找到掘金队（DEN）的列索引
    denver_column_index = df.columns.get_loc("DEN")-1

    # 寻找湖人队（Los Angeles Lakers）的行索引
    # 假设"Team"列包含球队名称
    lakers_row_index = df[df['Rk'] == 'Los Angeles Lakers'].index[0]

    # 根据找到的索引获取比分
    lakers_vs_nuggets_score = df.iloc[lakers_row_index, denver_column_index]

    print(f"湖人队掘金历史比分: {lakers_vs_nuggets_score}")
```

## 历史上湖人队的赛季胜率变化
```python
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('./湖人历史五赛季胜率.csv')

# 显示数据框
print(df)
```

## 通奸简单的线性回归检查湖人队的整体发展趋势
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取当前赛季的胜率数据
current_data = pd.read_csv('./current_win_rate.csv')

# 提取湖人队的胜率，保留两位小数
lakers_current_rate = round(current_data.loc[current_data['Team'] == 'Los Angeles Lakers', 'Win_Rate'].iloc[0], 2)

# 读取湖人历史五赛季的胜率数据
historical_data = pd.read_csv('./湖人历史五赛季胜率.csv')

# 提取历史胜率数据并转换为列表，假设历史数据的顺序正确（从最近到最远）
historical_rates = historical_data.iloc[0, 1:].tolist()  # 选取第一行，去掉'Team'列
historical_rates = [round(float(rate), 2) for rate in historical_rates]

# 将当前赛季的胜率添加到历史数据列表的开头
historical_rates.insert(0, lakers_current_rate)

# 创建X（年份，需要转换为numpy数组格式）和y（胜率）
years = list(range(2023, 2023 - len(historical_rates), -1))  # 动态生成年份列表
X = np.array([[year] for year in years])
y = np.array(historical_rates)

# 使用numpy进行线性回归：theta = (X^T X)^(-1) X^T y
X_b = np.c_[np.ones((len(X), 1)), X]  # 添加x0 = 1到每个实例
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# 预测所有年份的胜率
y_predict = X_b.dot(theta_best)

# 计算MSE
mse = np.mean((y - y_predict) ** 2)
print("MSE：:", mse)

# 显示实际与预测结果
print("实际:", y)
print("预测胜率:", [round(rate, 2) for rate in y_predict])

# 绘图展示实际与预测胜率
plt.figure(figsize=(10, 5))
plt.plot(years, y, 'o-', label='Actual Win Rate')
plt.plot(years, y_predict, 's--', label='Predicted Win Rate')
plt.title('Los Angeles Lakers Win Rate: Actual vs Predicted')
plt.xlabel('Year')
plt.ylabel('Win Rate')
plt.xticks(years, labels=[str(year) for year in years])
plt.legend()
plt.grid(True)
plt.show()



```

```{bibliography}
```

## Learn more

This is just a simple starter to get you started.
You can learn a lot more at [jupyterbook.org](https://jupyterbook.org).
