from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, GlobalAvgPool2D, Activation, Dense, Input, MaxPooling2D, Flatten
)
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

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
# VGG16 模块
# ----------------- #
def VGG16(x, num_classes=1000):
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # 全连接层
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    return x

# ----------------- #
# 使用 VGG16 模型
# ----------------- #
def create_VGG16_model(input_shape, num_classes=1000):
    inputs = Input(shape=input_shape)
    output = VGG16(inputs, num_classes=num_classes)
    model = Model(inputs, output)
    return model

# ----------------- #
# VGG16 网络
# ----------------- #
def VGG16_network(x, num_classes=1000):
    x = VGG16(x, num_classes=num_classes)
    return x

# ----------------- #
# 创建 VGG16 模型
# ----------------- #
def create_VGG16_model(input_shape, num_classes=1000):
    inputs = Input(shape=input_shape)
    output = VGG16_network(inputs, num_classes=num_classes)
    model = Model(inputs, output)
    return model

if __name__ == '__main__':
    inputs = Input(shape=(224, 224, 3))  # 假设输入形状为 (224, 224, 3) 适用于 VGG16
    model = create_VGG16_model(inputs.shape[1:], num_classes=1)  # 假设进行回归，有 1 个输出节点
    model.summary()

    # 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Reshape data for VGG16 input
    train_images_reshaped = train_images.reshape(train_images.shape[0], 224, 224, 4)
    validation_images_reshaped = validation_images.reshape(validation_images.shape[0], 224, 224, 4)

    # Normalize data
    train_x_reshaped = train_images_reshaped / 255.0
    validation_x_reshaped = validation_images_reshaped / 255.0

    # Train the model
    patience = 10
    # early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=2)
    hist = model.fit(train_images_reshaped, train_y, epochs=200, batch_size=32,
                     validation_data=(validation_images_reshaped, validation_y),
                     verbose=2, shuffle=True)

    from tensorflow.keras.optimizers import Adam

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

    # 创建一个DataFrame来存储评估指标
    evaluation_df = pd.DataFrame({
        'Epoch': range(1, len(loss_history) + 1),
        'Training Loss': loss_history,
        'Validation Loss': val_loss_history,
        'Training MAE': mae_history,
        'Validation MAE': val_mae_history,
        # 'Training MSE': mse_history,
        # 'Training MAE2': mae2_history,
    })

    # 将DataFrame保存到Excel文件
    with pd.ExcelWriter('evaluation_metrics2.xlsx', engine='xlsxwriter') as writer:
        evaluation_df.to_excel(writer, sheet_name='Evaluation Metrics', index=False)
        # 如果需要，您可以添加更多的工作表以存储其他数据

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

    plt.tight_layout()
    plt.show()



