import numpy as np
import pandas as pd
import argparse
import os
import sys
from model import *
from sklearn.metrics import confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser()
parser.add_argument('--number_filters', default=6, type=int)
parser.add_argument('--number_columns', default=14, type=int)
parser.add_argument('--filter_size', default=3, type=int, help='total number of training examples in a single batch')
parser.add_argument('--test_data', required=True)
#parser.add_argument('--structure', required=True)
parser.add_argument('--fold', type=int)
parser.add_argument('--folddir', required=True)
parser.add_argument('--outdir', required=True)
parser.add_argument('--type', required=True, type=str)

args = parser.parse_args()

test_data_0 = args.test_data
number_filters = args.number_filters
number_columns = args.number_columns
filter_size = args.filter_size
fold = args.fold
#feature = args.feature
outdir_0 = args.folddir
outdir_1 = args.outdir
#structure = args.structure

if os.path.exists(outdir_0):
   pass
else:
    os.makedirs(outdir_0,exist_ok=True)

data_type = args.type
length = 346 if data_type in ['BV', 'BY'] else None

# 构造 log 文件路径
log_path = os.path.join(outdir_0, str(fold), 'log.csv')
info = pd.read_csv(log_path, index_col=0)
info = info.iloc[:200]
dfmax = info[info['AUC'] == info['AUC'].max()].index[0] + 1
dfmax = f"{dfmax:02d}"  # Zero-pad to ensure two digits
# 假设你的模型保存在每个fold的子目录中
fold_dir = os.path.join(outdir_0, str(fold))
model_filename = f'model_{dfmax}'
model_path = os.path.join(fold_dir, model_filename)

#model_path = os.path.join(outdir_0, f'model_{dfmax}')
print(model_path)

####input the unseen data############################################
valid_data = np.load(test_data_0)
test_data=valid_data
#np.random.seed(100)
#shuffled_index = np.random.permutation(len(a))
#splits = np.array_split(shuffled_index, 5)
#test_index = splits[fold - 1]
#train_index = np.concatenate([splits[i] for i in range(5) if i != (fold - 1)], axis=0)
#a=np.load(test_data_0)#test_data=np.load(test_data_0)

##split 5




test_x = test_data[:, :, :number_columns]
test_y_0 = test_data[:, 0, number_columns]
test_y = np.zeros((test_x.shape[0], 2))

for i in range(test_x.shape[0]):
    test_y[i, 0] = 1 - test_y_0[i]
    test_y[i, 1] = test_y_0[i]

cnn = CNN(number_filters, filter_size, number_columns, length)
cnn.load_weights(model_path)
pred_y = cnn.predict(test_x)

pred_y_0 = pred_y[:, 1]
pred_y_0_np = np.array(pred_y_0)
true_y_np = np.array(test_y_0)
all_np = np.array(list(zip(true_y_np, pred_y_0_np)))
 ###计算混淆矩阵
cm = confusion_matrix(test_y_0, pred_y_0 > 0.5)  # 假设阈值为 0.5
np.save(os.path.join(outdir_1, f'confusion_matrix_{fold}.npy'), cm)
np.save(os.path.join(outdir_1, f'{fold}_pred.npy'), all_np)

os._exit(0)
sys.exit(0)
