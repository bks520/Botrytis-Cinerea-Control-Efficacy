#%%

import os
os.chdir('../')
import pandas as pd
import numpy as np
import transtab
# 指定保存嵌入结果的文件夹路径
output_folder = "C:\\Users\\zxs\\PycharmProjects\\transtab-main\\embedding_output"

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# set random seed
transtab.random_seed(42)

#%%

# load a dataset and start vanilla supervised training
allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = transtab.load_data('credit-g')

#%%

# make a fast pre-train of TransTab contrastive learning model
# build contrastive learner, set supervised=True for supervised VPCL
model, collate_fn = transtab.build_contrastive_learner(
    cat_cols, num_cols, bin_cols,
    supervised=True, # if take supervised CL
    num_partition=4, # num of column partitions for pos/neg sampling
    overlap_ratio=0.5, # specify the overlap ratio of column partitions during the CL
)

# start contrastive pretraining training
training_arguments = {
    'num_epoch':50,
    'batch_size':64,
    'lr':1e-4,
    'eval_metric':'val_loss',
    'eval_less_is_better':True,
    'output_dir':'./checkpoint'
    }

transtab.train(model, trainset, valset, collate_fn=collate_fn, **training_arguments)

#%%

# There are two ways to build the encoder
# First, take the whole pretrained model and output the cls token embedding at the last layer's outputs
enc = transtab.build_encoder(
    binary_columns=bin_cols,
    checkpoint = './checkpoint'
)

#%%

# Then take the encoder to get the input embedding
df = trainset[0]
output = enc(df)
print(output.shape)
output[:2]

#%%

df.head()

#%%

# Second, if we only want to the embeded token level embeddings (embeddings before going to transformers)
enc = transtab.build_encoder(
    binary_columns=bin_cols,
    checkpoint='./checkpoint'
)

# 获取嵌入结果
df = trainset[0]
output = enc(df)

# 将张量移动到 CPU 并转换为 NumPy 数组
output = output.cpu().detach().numpy()

# 生成文件路径
output_file = os.path.join(output_folder, "embedding.npy")

# 保存嵌入结果为 NumPy 数组文件
np.save(output_file, output)
# 加载 .npy 文件
data = np.load("C:\\Users\\zxs\\PycharmProjects\\transtab-main\\embedding_output\\embedding.npy")

# 将数据转换为 DataFrame
df = pd.DataFrame(data)

# 保存为 Excel 文件
output_file = "C:\\Users\\zxs\\PycharmProjects\\transtab-main\\embedding_output\\output.xlsx"
df.to_excel(output_file, index=False)