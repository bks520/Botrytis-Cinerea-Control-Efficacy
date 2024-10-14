from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, GlobalAvgPool2D, Activation, Multiply,
    Add, Dense, Input
)
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
tf.autograph.set_verbosity(0)

from tensorflow.keras.layers import Conv2D, Multiply, Add, GlobalAveragePooling2D, Reshape, Permute
from tensorflow.keras.layers import Conv2D, Multiply, Add, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Permute

from tensorflow.keras.layers import Concatenate
import shap
from sklearn.model_selection import train_test_split
import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential  # 确保你从正确的模块导入了Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer

# ...（你的代码的其余部分）





# ----------------- #
# 卷积+标准化
# ----------------- #
def conv_bn(filters, kernel_size, strides, padding, groups=1):
    def _conv_bn(x):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                   padding=padding, groups=groups, use_bias=False)(x)
        x = BatchNormalization()(x)
        return x

    return _conv_bn

# ----------------- #
# CBAM模块
# ----------------- #
def SE_MHSA_block(x, se_r=16, num_heads=8):
    # SE模块
    se_channels = x.shape[-1]
    se = GlobalAvgPool2D()(x)
    se = Reshape((1, 1, se_channels))(se)
    se = Conv2D(filters=se_channels // se_r, kernel_size=1, strides=1)(se)
    se = Activation('relu')(se)
    se = Conv2D(filters=se_channels, kernel_size=1, strides=1)(se)
    se = Activation('sigmoid')(se)
    x_se = Multiply()([x, se])

    # Multi-Head Self-Attention模块
    channels = x.shape[-1]
    head_dim = channels // num_heads
    heads = []
    for i in range(num_heads):
        head = Conv2D(filters=head_dim, kernel_size=1, strides=1, padding='same')(x)
        heads.append(head)

    f = Conv2D(filters=head_dim, kernel_size=1, strides=1, padding='same')(x)
    g = Conv2D(filters=head_dim, kernel_size=1, strides=1, padding='same')(x)
    h = Conv2D(filters=channels, kernel_size=1, strides=1, padding='same')(x)

    s = Multiply()([g, h])
    beta = Add()([f, s])
    beta = Activation('softmax')(beta)

    attended_heads = []
    for i in range(num_heads):
        attended_head = Multiply()([beta, heads[i]])
        attended_head = Conv2D(filters=channels // num_heads, kernel_size=1, strides=1, padding='same')(attended_head)
        attended_heads.append(attended_head)

    attended_features = Add()(attended_heads)
    x_mhsa = Add()([x, attended_features])

    # 合并SE和Multi-Head Self-Attention模块
    x_combined = Add()([x_se, x_mhsa])

    return x_combined

# 导入你的 RepVGGBlock 函数
# 导入你的 RepVGGBlock 函数
def RepVGGBlock(filters, kernel_size, strides=1, padding='valid', dilation=1, groups=1, deploy=False, use_se=False):
    class CustomLayer(Layer):
        def __init__(self, **kwargs):
            super(CustomLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            super(CustomLayer, self).build(input_shape)

        def call(self, inputs):
            if deploy:
                if use_se:
                    x = SE_MHSA_block(inputs)  # 使用CBAM模块替代SE模块
                    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding=padding, dilation_rate=dilation, groups=groups, use_bias=True)(x)
                    x = Activation('relu')(x)
                else:
                    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding=padding, dilation_rate=dilation, groups=groups, use_bias=True)(inputs)
                    x = SE_MHSA_block(x)  # 使用CBAM模块替代SE模块
                    x = Activation('relu')(x)
                return x

            if inputs.shape[-1] == filters and strides == 1:
                if use_se:
                    id_out = BatchNormalization()(inputs)
                    x1 = conv_bn(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, groups=groups)(
                        inputs)
                    x2 = conv_bn(filters=filters, kernel_size=1, strides=strides, padding=padding, groups=groups)(inputs)
                    x3 = Add()([id_out, x1, x2])
                    x4 = SE_MHSA_block(x3)
                    return Activation('relu')(x4)
                else:
                    id_out = BatchNormalization()(inputs)
                    x1 = conv_bn(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, groups=groups)(
                        inputs)
                    x2 = conv_bn(filters=filters, kernel_size=1, strides=strides, padding=padding, groups=groups)(inputs)
                    x3 = Add()([id_out, x1, x2])
                    return Activation('relu')(x3)
            else:
                if use_se:
                    x1 = conv_bn(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, groups=groups)(
                        inputs)
                    x2 = conv_bn(filters=filters, kernel_size=1, strides=strides, padding='valid', groups=groups)(inputs)
                    x3 = Add()([x1, x2])
                    x4 = SE_MHSA_block(x3)
                    return Activation('relu')(x4)
                else:
                    x1 = conv_bn(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, groups=groups)(
                        inputs)
                    x2 = conv_bn(filters=filters, kernel_size=1, strides=strides, padding='valid', groups=groups)(inputs)
                    x3 = Add()([x1, x2])
                    return Activation('relu')(x3)

    return CustomLayer()
# 读取数据
data = pd.read_csv('C:\\Users\\zxs\\PycharmProjects\\transtab-main\\shujujizxs2.csv')  # 请替换为你的数据文件路径

# 分割数据集为特征和目标
X = data[['nongdu', 'guangzhao', 'wendu', 'yingyang']]
y = data['mianji']

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建一个深度学习模型，将 RepVGGBlock 集成进去
model = Sequential()
input_shape = (height, width, channels)  # 替换成实际的图像维度
model.add(Input(shape=input_shape))


# 在模型中使用 RepVGGBlock，根据需要添加多个
model.add(RepVGGBlock(filters=64, kernel_size=(3, 3), strides=1, padding='valid', dilation=1, groups=1, use_se=True))
model.add(RepVGGBlock(filters=128, kernel_size=(3, 3), strides=1, padding='valid', dilation=1, groups=1, use_se=True))

# 添加其他层，例如全连接层或输出层，适应你的任务

# 编译模型
model.compile(optimizer='adam', loss='mse')  # 使用适当的优化器和损失函数

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32)  # 根据需要调整 epochs 和 batch_size

# 创建一个SHAP解释器
explainer = shap.Explainer(model, X_train)

# 计算SHAP值
shap_values = explainer.shap_values(X_test)

# 可视化SHAP值，并传入特征名称
shap.summary_plot(shap_values, X_test, feature_names=X.columns)
