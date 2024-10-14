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





# Load and preprocess data
df = pd.read_excel('C:\\Users\\zxs\\PycharmProjects\\transtab-main\\shujujizxs2.xlsx', sheet_name=0)
data = df.iloc[:, 1:6].values
scale = np.amax(data[:, -1])
sep = int(0.75 * len(data))

# Split the data into train and validation sets
train_x = data[:sep, :5]
train_y = data[:sep, -1] / scale
validation_x = data[sep:, :5]
validation_y = data[sep:, -1] / scale

# ----------------- #
# 图像数据处理
# ----------------- #
def convert_to_images(data, scale):
    image_height = 224
    image_width = 224
    images = []

    for row in data:
        image = np.zeros((image_height, image_width, 4), dtype=np.uint8)
        for i in range(4):  # 仅使用前4个特征为RGB通道赋值
            pixel_value = int(row[i] * 255)
            image[:, :, i] = pixel_value
        intensity_value = int(row[-1] / scale * 255)
        image[:, :, 3] = intensity_value  # 最后一个通道表示强度值
        images.append(image)

    return np.array(images)

train_images = convert_to_images(train_x, scale)
validation_images = convert_to_images(validation_x, scale)

# Normalize data
train_images = train_images / 255.0
validation_images = validation_images / 255.0

# # 绘制样本图像
# sample_index = 0  # 更改为您想要显示的样本索引
# sample_image = train_images[sample_index]
# plt.imshow(sample_image)
# plt.title(f"Sample Image - Label: {train_y[sample_index] * scale:.2f}")
# plt.show()


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
def MultiHeadSelfAttention_block(x, num_heads=8):
    channels = x.shape[-1]

    head_dim = channels // num_heads

    # 分割输入特征图为不同的头
    heads = []
    for i in range(num_heads):
        head = Conv2D(filters=head_dim, kernel_size=1, strides=1, padding='same')(x)
        heads.append(head)

    # 计算注意力权重
    f = Conv2D(filters=head_dim, kernel_size=1, strides=1, padding='same')(x)
    g = Conv2D(filters=head_dim, kernel_size=1, strides=1, padding='same')(x)
    h = Conv2D(filters=channels, kernel_size=1, strides=1, padding='same')(x)

    s = Multiply()([g, h])
    beta = Add()([f, s])
    beta = Activation('softmax')(beta)

    # 计算自注意力特征
    attended_heads = []
    for i in range(num_heads):
        attended_head = Multiply()([beta, heads[i]])
        attended_head = Conv2D(filters=channels // num_heads, kernel_size=1, strides=1, padding='same')(attended_head)
        attended_heads.append(attended_head)

    # 合并所有头
    attended_features = Add()(attended_heads)

    # 添加残差连接
    x = Add()([x, attended_features])

    return x


# ----------------- #
# RepVGG模块的堆叠
# ----------------- #
def make_stage(planes, num_blocks, stride_1, deploy, use_se, override_groups_map=None):
    def _make_stage(x):
        cur_layer_id = 1
        strides = [stride_1] + [1] * (num_blocks - 1)
        for stride in strides:
            cur_groups = override_groups_map.get(cur_layer_id, 1)
            x = RepVGGBlock(filters=planes, kernel_size=3, strides=stride, padding='same',
                            groups=cur_groups, deploy=deploy, use_se=use_se)(x)
            cur_layer_id += 1
        return x

    return _make_stage

# ----------------- #
# RepVGG模块
# ----------------- #
def RepVGGBlock(filters, kernel_size, strides=1, padding='valid', dilation=1, groups=1, deploy=False, use_se=False):
    def _RepVGGBlock(inputs):
        if deploy:
            if use_se:
                x = MultiHeadSelfAttention_block(inputs)  # 使用CBAM模块替代SE模块
                x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                           padding=padding, dilation_rate=dilation, groups=groups, use_bias=True)(x)
                x = Activation('relu')(x)
            else:
                x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                           padding=padding, dilation_rate=dilation, groups=groups, use_bias=True)(inputs)
                x = MultiHeadSelfAttention_block(x)  # 使用CBAM模块替代SE模块
                x = Activation('relu')(x)
            return x

        # 其余部分保持不变


        # 其余部分保持不变


        if inputs.shape[-1] == filters and strides == 1:
            if use_se:
                id_out = BatchNormalization()(inputs)
                x1 = conv_bn(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, groups=groups)(
                    inputs)
                x2 = conv_bn(filters=filters, kernel_size=1, strides=strides, padding=padding, groups=groups)(inputs)
                x3 = Add()([id_out, x1, x2])
                x4 = MultiHeadSelfAttention_block(x3)
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
                x4 = MultiHeadSelfAttention_block(x3)
                return Activation('relu')(x4)
            else:
                x1 = conv_bn(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, groups=groups)(
                    inputs)
                x2 = conv_bn(filters=filters, kernel_size=1, strides=strides, padding='valid', groups=groups)(inputs)
                x3 = Add()([x1, x2])

                return Activation('relu')(x3)

    return _RepVGGBlock






# ----------------- #
# RepVGG网络
# ----------------- #
def RepVGG(x, num_blocks, classes=1000, width_multiplier=None, override_groups_map=None, deploy=False, use_se=False):
    override_groups_map = override_groups_map or dict()
    in_planes = min(64, int(64 * width_multiplier[0]))
    out = RepVGGBlock(filters=in_planes, kernel_size=3, strides=2, padding='same', deploy=deploy, use_se=use_se)(x)
    out = make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride_1=2, deploy=deploy, use_se=use_se,
                     override_groups_map=override_groups_map)(out)
    out = make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride_1=2, deploy=deploy, use_se=use_se,
                     override_groups_map=override_groups_map)(out)
    out = make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride_1=2, deploy=deploy, use_se=use_se,
                     override_groups_map=override_groups_map)(out)
    out = make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride_1=2, deploy=deploy, use_se=use_se,
                     override_groups_map=override_groups_map)(out)
    out = GlobalAvgPool2D()(out)
    out = Dense(classes)(out)
    return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


def RepVGG_A0(inputs, classes=1000, deploy=False):
    return RepVGG(inputs, num_blocks=[2, 4, 14, 1], classes=classes,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)


def create_RepVGG_A1(x, deploy=False):
    return RepVGG(x, num_blocks=[2, 4, 14, 1], classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)


def create_RepVGG_A2(x, deploy=False):
    return RepVGG(x, num_blocks=[2, 4, 14, 1], classes=1000,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy)


def create_RepVGG_B0(x, deploy=False):
    return RepVGG(x, num_blocks=[4, 6, 16, 1], classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)


def create_RepVGG_B1(x, deploy=False):
    return RepVGG(x, num_blocks=[4, 6, 16, 1], classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy)


def create_RepVGG_B1g2(x, deploy=False):
    return RepVGG(x, num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B1g4(x, deploy=False):
    return RepVGG(x, num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B2(x, deploy=False):
    return RepVGG(x, num_blocks=[4, 6, 16, 1], classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy)


def create_RepVGG_B2g2(x, deploy=False):
    return RepVGG(x, num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B2g4(x, deploy=False):
    return RepVGG(x, num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B3(x, deploy=False):
    return RepVGG(x, num_blocks=[4, 6, 16, 1], classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy)


def create_RepVGG_B3g2(x, deploy=False):
    return RepVGG(x, num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B3g4(x, deploy=False):
    return RepVGG(x, num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_D2se(x, deploy=False):
    return RepVGG(x, num_blocks=[8, 14, 24, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_se=True)


if __name__ == '__main__':
    inputs = Input(shape=(1, 1, 5))

    classes = 1000
    model = Model(inputs=inputs, outputs=create_RepVGG_A1(inputs))
    model.summary()


    # 定义 RMSE 和 R^2 指标
    def rmse(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


    def r2(y_true, y_pred):
        SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
        SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())


    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse', rmse, r2])

# Reshape data for RepVGG input
train_x_reshaped = train_x.reshape(train_x.shape[0], 1, 1, 5)
validation_x_reshaped = validation_x.reshape(validation_x.shape[0], 1, 1, 5)



# Normalize data
train_x_reshaped = train_x_reshaped / 255.0
validation_x_reshaped = validation_x_reshaped / 255.0

import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 定义 RMSE 和 R^2 指标
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def r2(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Train the model
patience = 10
# early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=2)
hist = model.fit(train_x_reshaped, train_y, epochs=200, batch_size=32, validation_data=(validation_x_reshaped, validation_y),
                 verbose=2, shuffle=True, )
from tensorflow.keras.optimizers import Adam

learning_rate = 0.000001  # 设置您想要的学习率
optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# callbacks=[early_stopping]
from tensorflow.keras.optimizers import Adam
# 在验证集上评估模型
val_loss, val_mae, val_mse, val_rmse, val_r2 = model.evaluate(validation_x_reshaped, validation_y, verbose=0)
print(f'验证损失: {val_loss:.4f}')
print(f'验证MAE: {val_mae:.4f}')
print(f'验证MSE: {val_mse:.4f}')
print(f'验证RMSE: {val_rmse:.4f}')
print(f'验证R^2: {val_r2:.4f}')

# 预测并计算其他指标
val_predictions = model.predict(validation_x_reshaped)
rmse = np.sqrt(mean_squared_error(validation_y, val_predictions))
r2 = r2_score(validation_y, val_predictions)

print(f'验证RMSE: {rmse:.4f}')
print(f'验证R^2: {r2:.4f}')

learning_rate = 0.000001  # 设置您想要的学习率
optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# callbacks=[early_stopping]




import matplotlib.pyplot as plt

# print("Validation Ground Truth Labels:")
# print(train_y)


# Get loss and mae history
loss_history = hist.history['loss']
val_loss_history = hist.history['val_loss']
mae_history = hist.history['mae']
val_mae_history = hist.history['val_mae']

import pandas as pd

# 创建一个空的DataFrame来存储每轮的训练结果
training_results = pd.DataFrame(columns=['Epoch', 'Loss', 'MAE', 'MSE', 'RMSE', 'R^2'])

# 定义训练循环
num_epochs = 5  # 根据您的需求设置训练轮数
for epoch in range(num_epochs):
    # 训练模型（这里使用你的训练代码）

    # 在验证集上评估模型
    evaluation_results = model.evaluate(validation_x_reshaped, validation_y, verbose=0)
    val_loss = evaluation_results[0]
    val_mse = evaluation_results[1]

    # 打印并保存每轮的结果
    print(f'Epoch {epoch + 1}:')
    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Validation MAE: {val_mae:.4f}')
    print(f'Validation MSE: {val_mse:.4f}')
    print(f'Validation RMSE: {val_rmse:.4f}')
    print(f'Validation R^2: {val_r2:.4f}')

    # 将结果添加到DataFrame
    training_results = training_results.append({
        'Epoch': epoch + 1,
        'Loss': val_loss,
        'MAE': val_mae,
        'MSE': val_mse,
        'RMSE': val_rmse,
        'R^2': val_r2
    }, ignore_index=True)

# 保存结果到Excel文件
training_results.to_excel('training_results.xlsx', index=False)





# Plot loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(loss_history, label='Training Loss')
# plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(mae_history, label='Training MAE')
# plt.plot(val_mae_history, label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Training and Validation MAE')
plt.legend()

# Plot RMSE and R^2
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(rmse, label='RMSE', color='red')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Root Mean Squared Error (RMSE)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(r2, label='R^2', color='green')
plt.xlabel('Epoch')
plt.ylabel('R^2')
plt.title('R-Squared (R^2)')
plt.legend()

plt.tight_layout()
plt.show()





