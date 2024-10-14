import pandas as pd

import transtab

# set random seed
transtab.random_seed(42)

# load a dataset and start vanilla supervised training
allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
    = transtab.load_data('credit-g')

data_frame=pd.read_excel(r'C:\\Users\\zxs\\PycharmProjects\\transtab-main\\shujuji111.xlsx')
input=data_frame.loc[:,"浓度梯度":"营养"]
lab=data_frame.iloc[:,5]
tup=input,lab
# print(tup)
trainset=[]
valset=[]
testset=[]
trainset.append(tup)
valset.append(tup)
testset.append(tup)
cat_cols=None
bin_cols=None
num_cols=['浓度梯度','光照','温度','营养']
# print(bin_cols)
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
    'num_epoch':5,
    'batch_size':64,
    'lr':1e-4,
    'eval_metric':'val_loss',
    'eval_less_is_better':True,
    'output_dir':'./checkpoint'
    }

transtab.train(model, trainset, valset, collate_fn=collate_fn, **training_arguments)

# There are two ways to build the encoder
# First, take the whole pretrained model and output the cls token embedding at the last layer's outputs
enc = transtab.build_encoder(
    binary_columns=bin_cols,
    checkpoint = './checkpoint'
)

# Then take the encoder to get the input embedding
df = trainset[0][0]        #
# print(df)
output = enc(df)
print(output)
output[:2]

df.head()

# Second, if we only want to the embeded token level embeddings (embeddings before going to transformers)
enc = transtab.build_encoder(
    binary_columns=bin_cols,
    checkpoint = './checkpoint',
    num_layer = 0,
)

output = enc(df)
print(output['embedding'].shape)
output['embedding'][:2]

