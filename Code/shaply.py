import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt  # 添加用于绘制图形的库

# 读取数据
data = pd.read_csv('C:\\Users\\zxs\\PycharmProjects\\transtab-main\\shujujizxs2.csv')  # 请替换为你的数据文件路径

# 分割数据集为特征和目标
X = data[['nongdu', 'guangzhao', 'wendu', 'yingyang']]
y = data['mianji']

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练一个随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 创建一个SHAP解释器
explainer = shap.Explainer(model, X_train)

# 计算SHAP值
shap_values = explainer.shap_values(X_test)

# # 可视化SHAP值，并传入特征名称
# shap.summary_plot(shap_values, X_test, feature_names=X.columns)

# 选择一个要解释的样本（这里选择第一个样本）
sample_index = 0

# # 创建SHAP瀑布图
# shap.plots.waterfall(shap.Explanation(values=shap_values[sample_index], base_values=explainer.expected_value, data=X_test.iloc[sample_index], feature_names=X.columns))

# 显示图形
plt.show()

# 可视化SHAP值，使用条形图
shap.summary_plot(shap_values, X_test, feature_names=X.columns, plot_type="bar")

