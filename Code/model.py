import torch
import torch.nn as nn
import torch.optim as optim
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

from VGGnet import train_x
from VGGnet2 import train_y

tf.autograph.set_verbosity(0)

from tensorflow.keras.layers import Conv2D, Multiply, Add, GlobalAveragePooling2D, Reshape, Permute
from tensorflow.keras.layers import Conv2D, Multiply, Add, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Permute

# 定义您的 PyTorch 模型类
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

# 创建并训练您的 PyTorch 模型，使用您的数据和其他参数
model = SE_MHSA_block()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 将数据转换为 PyTorch 张量
train_x_tensor = torch.Tensor(train_x)
train_y_tensor = torch.Tensor(train_y)

# 训练模型
epochs = 100
for epoch in range(epochs):
    # 前向传播
    outputs = model(train_x_tensor)
    loss = criterion(outputs, train_y_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
