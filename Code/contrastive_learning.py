#%%

import os
os.chdir('../')

import transtab

# set random seed
transtab.random_seed(42)

#%%

# load multiple datasets by passing a list of data names
allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
    = transtab.load_data(['credit-g','credit-approval'])

# build contrastive learner, set supervised=True for supervised VPCL
model, collate_fn = transtab.build_contrastive_learner(
    cat_cols, num_cols, bin_cols,
    supervised=True, # if take supervised CL
    num_partition=4, # num of column partitions for pos/neg sampling
    overlap_ratio=0.5, # specify the overlap ratio of column partitions during the CL
)

#%%

# start contrastive pretraining training
training_arguments = {
    'num_epoch':20,
    'batch_size':64,
    'lr':1e-4,
    'eval_metric':'val_loss',
    'eval_less_is_better':True,
    'output_dir':'./checkpoint'
    }

transtab.train(model, trainset, valset, collate_fn=collate_fn, **training_arguments)

#%%

# load the pretrained model and finetune on a target dataset
allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
     = transtab.load_data('credit-approval')

# build transtab classifier model, and load from the pretrained dir
model = transtab.build_classifier(checkpoint='./checkpoint')

# update model's categorical/numerical/binary column dict
model.update({'cat':cat_cols,'num':num_cols,'bin':bin_cols})

# start finetuning
training_arguments = {
    'num_epoch':50,
    'eval_metric':'val_loss',
    'eval_less_is_better':True,
    'output_dir':'./checkpoint'
    }
transtab.train(model, trainset, valset, **training_arguments)

#%%

# evaluation
x_test, y_test = testset
ypred = transtab.predict(model, x_test)
transtab.evaluate(ypred, y_test, metric='auc')

#%%


