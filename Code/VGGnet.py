from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, add
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import numpy as np
from tensorflow.keras.layers import BatchNormalization

import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2  # 这里加入l2正则化目的是为了防止过拟合
from tensorflow.keras.callbacks import EarlyStopping
import xlrd
import pandas as pd
import time
import matplotlib

#matplotlib.rcParams['font.family'] = 'sans-serif'
from resnet50zxs import test_x

matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'# 中文设置成宋体，除此之外的字体设置成New Roman

start = time.time()
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

x_num = 205#x特征数
y_col = 208#y在excel表中的列数，含第0列为样点编号列
df=pd.read_excel('C:\\Users\\zxs\\PycharmProjects\\transtab-main\\R.xlsx',sheet_name=0)#可以通过表单索引来指定读取第一个sheet表单
data = df.iloc[:,1:y_col].values#读取前206列所有行的值,并将数据转为numpy数据
print (data.shape)
print(data[:9,-1])
scale = np.amax(data[:,y_col-2])#所有y中最大的值
print (scale)
sep = int(0.75*len(data))#设置训练和测试数据集比例，180个样点训练，65个测试
train = data[:sep]
validation = data[sep:]
train_x = train[:,:x_num]
train_x = train_x.reshape(len(train_x[:,0]),len(train_x[0,:]),1)#加入通道为1
train_y = train[:,y_col-2]
train_y = np.divide(train_y,scale)#归一化y值
validation_x = validation[:,:x_num]
validation_x = validation_x.reshape(len(validation_x[:,0]),len(validation_x[0,:]),1)#加入通道为1
validation_y = validation[:,y_col-2]
validation_y = np.divide(validation_y,scale)#归一化y值
print(train_x.shape)
print(validation_x.shape)

def coeff_determination(y_true, y_pred):
      SS_res = K.sum(K.square( y_true*scale-y_pred*scale ))
      SS_tot = K.sum(K.square( y_true*scale - K.mean(y_true*scale) ))
      return 1 - SS_res/(SS_tot + K.epsilon())
def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred*scale - y_true*scale)))
def rpd(y_true, y_pred):
        return K.sqrt(K.mean(K.square( y_true*scale-K.mean(y_true*scale))))/K.sqrt(K.mean(K.square(y_pred*scale - y_true*scale)))

if K.image_data_format() == 'channels_first':
        train_x = train_x.reshape(train_x.shape[0], 1, 1, x_num)
        validation_x = validation_x.reshape(validation_x.shape[0], 1, 1, x_num)
        input_shape = (1, 1, x_num)
else:
        train_x = train_x.reshape(train_x.shape[0], 1, x_num, 1)
        validation_x = validation_x.reshape(validation_x.shape[0], 1, x_num, 1)
        input_shape = (1, x_num, 1)#input_shape = (1, 205, 1)

nadam = optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

model = Sequential()
model.add(Conv2D(6,(1,3),strides=(1,1),input_shape=input_shape,padding='same',activation='tanh'))
model.add(Conv2D(6,(1,3),strides=(1,1),padding='same',activation='tanh'))
model.add(MaxPooling2D(pool_size=(1,2),strides=(1,2)))

model.add(Conv2D(12,(1,3),strides=(1,1),padding='same',activation='tanh'))
model.add(Conv2D(12,(1,3),strides=(1,1),padding='same',activation='tanh'))
model.add(MaxPooling2D(pool_size=(1,2),strides=(1,2)))

model.add(Flatten())
model.add(Dense(200,activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(100,activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.summary()
patience=10
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=2)
model.compile(optimizer=nadam,loss=rmse,metrics=[coeff_determination,rmse,rpd])
hist = model.fit(train_x,train_y,epochs=10,validation_data=(validation_x, validation_y),verbose=2, shuffle=False, callbacks=[early_stopping])
preds = model.predict(validation_x)*scale

import shap
import numpy as np

# select a set of background examples to take an expectation over
background = train_x[np.random.choice(train_x.shape[0], 100, replace=False)]

# explain predictions of the model on four images
e = shap.DeepExplainer(model, background)
# ...or pass tensors directly
e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
shap_values = e.shap_values(test_x)

# plot the feature attributions
shap.image_plot(shap_values, -test_x[1:5])
print(np.array(shap_values).shape)
#print(train_x.shape)
tests=np.array(shap_values)
import pandas as pd


# 画图代码
shap.summary_plot(shap_values, test_x) #输出特征重要性

#   注意 ！！ 该库的shap_values值为 分每个样本计算(每个样本中特征的重要性贡献程度可能不一样）
#，该图例的计算方式为  将每个值的shap_values取绝对值后平均后输出