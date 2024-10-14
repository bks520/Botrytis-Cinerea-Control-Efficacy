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

from timm.models import features

tf.autograph.set_verbosity(0)

from tensorflow.keras.layers import Conv2D, Multiply, Add, GlobalAveragePooling2D, Reshape, Permute
from tensorflow.keras.layers import Conv2D, Multiply, Add, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Permute
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split







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

        # 其余部分保持不变


        # 其余部分保持不变


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
    return RepVGG(x, num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy)


def create_RepVGG_B0(x, deploy=False):
    return RepVGG(x, num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)


def create_RepVGG_B1(x, deploy=False):
    return RepVGG(x, num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy)


def create_RepVGG_B1g2(x, deploy=False):
    return RepVGG(x, num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B1g4(x, deploy=False):
    return RepVGG(x, num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B2(x, deploy=False):
    return RepVGG(x, num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy)


def create_RepVGG_B2g2(x, deploy=False):
    return RepVGG(x, num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B2g4(x, deploy=False):
    return RepVGG(x, num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B3(x, deploy=False):
    return RepVGG(x, num_blocks=[4, 6, 16, 1], num_classes=1000,
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
    model = Model(inputs=inputs, outputs=RepVGG_A0(inputs))
    model.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Reshape data for RepVGG input
train_x_reshaped = train_x.reshape(train_x.shape[0], 1, 1, 5)
validation_x_reshaped = validation_x.reshape(validation_x.shape[0], 1, 1, 5)



# Normalize data
train_x_reshaped = train_x_reshaped / 255.0
validation_x_reshaped = validation_x_reshaped / 255.0

explainer = shap.KernelExplainer(model.predict,validation_x_reshaped)

shap_values = explainer.shap_values(validation_x_reshaped,nsamples=100)

shap.summary_plot(shap_values, validation_x_reshaped, feature_names=["浓度梯度", "光照", "温度","营养"])
