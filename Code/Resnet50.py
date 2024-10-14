from tensorflow.keras.layers import Input, Conv1D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping


# 加载和预处理数据（与之前相同）
df = pd.read_excel('C:\\Users\\zxs\\PycharmProjects\\transtab-main\\shujujizxs2.xlsx', sheet_name=0)
data = df.iloc[:, 1:6].values
scale = np.amax(data[:, -1])
sep = int(0.75 * len(data))
train = data[:sep]
validation = data[sep:]
train_x = train[:, :5]
train_y = train[:, -1] / scale
validation_x = validation[:, :5]
validation_y = validation[:, -1] / scale

# 创建一维卷积模型
input_layer = Input(shape=(5, 1))
conv1 = Conv1D(6, 3, padding='same', activation='tanh')(input_layer)
conv2 = Conv1D(12, 3, padding='same', activation='tanh')(conv1)
pooling = GlobalAveragePooling1D()(conv2)
dense1 = Dense(200, activation='tanh')(pooling)
dropout1 = Dropout(0.3)(dense1)
dense2 = Dense(100, activation='tanh')(dropout1)
dropout2 = Dropout(0.3)(dense2)
output_layer = Dense(1, activation='sigmoid')(dropout2)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004),
              loss='mean_squared_error',
              metrics=['mae'])
model.summary()




# Reshape data for 1D Convolution
train_x = train_x.reshape(train_x.shape[0], 5, 1)
validation_x = validation_x.reshape(validation_x.shape[0], 5, 1)

# Train the model
patience = 10
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=2)
hist = model.fit(train_x, train_y, epochs=100, batch_size=16, validation_data=(validation_x, validation_y),
                 verbose=2, shuffle=True, callbacks=[early_stopping])

# Predict and evaluate
preds = model.predict(validation_x) * scale

