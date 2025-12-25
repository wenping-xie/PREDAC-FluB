import numpy as np
import pandas
import argparse
from model import *
import sys
import os
from sklearn import model_selection
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import collections
###not use gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




parser = argparse.ArgumentParser()
parser.add_argument('--number_filters', default=6, type=int)
parser.add_argument('--number_columns',  type=int)
parser.add_argument('--filter_size', default=3, type=int, help='total number of training examples in a single batch')
parser.add_argument('--epoch',default=200,type=int)
parser.add_argument('--aaindexfile', required=True)
parser.add_argument('--fold', type=int)
parser.add_argument('--outdir',required=True)
parser.add_argument('--type',required=True)
args = parser.parse_args()


aaindexfile=args.aaindexfile
number_filters=args.number_filters
number_columns=args.number_columns
filter_size=args.filter_size
fold=args.fold
epoch=args.epoch


outdir_0=args.outdir+'/'+str(fold)
if os.path.exists(outdir_0):
    pass
else:
    os.makedirs(outdir_0)

data_type=args.type
if data_type=='BV':
    length=346
elif data_type=='BY':
    length=346

seed=100
file=aaindexfile
data = np.load(file)

print('data',collections.Counter(data[:,0,number_columns]))
train_and_test, valid = model_selection.train_test_split(data, test_size=0.1,random_state=seed, stratify=data[:,0,number_columns])  # 先分为两部分：训练和验证  ，  测试集
np.save(outdir_0+'/valid_data_BV.npy',valid)
np.save(outdir_0+'/train_and_test.npy',train_and_test)
print('train_and_valid_label ',collections.Counter(train_and_test[:,0,number_columns]))
print('testing_label',collections.Counter(valid[:,0,number_columns]))

sss = StratifiedShuffleSplit(n_splits=5, test_size=1/8, random_state=seed)
y=train_and_test[:,0,number_columns]

for fold, (train_index, test_index) in enumerate(sss.split(train_and_test,y),1):
    outdir_0=args.outdir+'/'+str(fold)
    if os.path.exists(outdir_0):
        pass
    else:
        os.makedirs(outdir_0)

    train_data = train_and_test[train_index]
    test_data = train_and_test[test_index]
    train_x_0=train_data[:,:,:number_columns]
    train_x_copy=np.roll(train_x_0,-int(number_columns/2),axis=2)
    train_x=np.concatenate((train_x_0,train_x_copy),axis=0)

    train_y_0_0=train_data[:,0,number_columns]
    train_y_0=np.concatenate((train_y_0_0,train_y_0_0))
    test_x=test_data[:,:,:number_columns]
    test_y_0=test_data[:,0,number_columns]
    train_y=np.zeros((train_x.shape[0],2))
    test_y=np.zeros((test_x.shape[0],2))
    for i in range(train_x.shape[0]):
        train_y[i,0]=1-train_y_0[i]
        train_y[i,1]=train_y_0[i]

    for i in range(test_x.shape[0]):
        test_y[i,0]=1-test_y_0[i]
        test_y[i,1]=test_y_0[i]

    checkpoiner=tf.keras.callbacks.ModelCheckpoint(filepath=outdir_0+'/model_{epoch:02d}',monitor='val_loss',save_weights_only=True,verbose=1)
    cnn = CNN(number_filters,filter_size,number_columns,length)
    history_callback=cnn.fit(
            x=train_x, y=train_y,
            batch_size=128,
            epochs=epoch,
            verbose=1,
            callbacks=[checkpoiner],
            shuffle=True,
            validation_data=(test_x,test_y))

    TP_all=[]
    FP_all=[]
    FN_all=[]
    TN_all=[]
    AUC_all=[]
    for i in range(epoch):
        i+=1
        i='%02d' %i
        cnn.load_weights(outdir_0+'/model_'+str(i))
        pred_y=cnn.predict(test_x)
        TP, FP, TN, FN=get_confusion_matrix(test_y, pred_y)
        #fpr, tpr, thresholds = roc_curve(test_y,pred_y)
        auc = roc_auc_score(test_y, pred_y)
        TP_all.append(TP)
        FP_all.append(FP)
        TN_all.append(TN)
        FN_all.append(FN)
        AUC_all.append(auc)

    new_dict=history_callback.history.copy()
    new_dict['TN']=TN_all
    new_dict['FP']=FP_all
    new_dict['FN']=FN_all
    new_dict['TP']=TP_all
    new_dict['AUC']=AUC_all




    pandas.DataFrame(new_dict).to_csv(outdir_0 + '/log.csv')

os._exit(0)
sys.exit(0)

