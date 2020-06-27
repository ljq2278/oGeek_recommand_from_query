import pandas as pd
import numpy as np
from sklearn.metrics import f1_score


# 由于无法线上测试了，直接拿验证集来做测试集
train_data = pd.read_table('E:/data/oppo_ogeek_query_match/oppo_round1_train_20180929.txt',names=['prefix','query_prediction','title','tag','label'],header=None,encoding='utf-8').astype(str)
test_data = pd.read_table('E:/data/oppo_ogeek_query_match/oppo_round1_vali_20180929.txt',names=['prefix','query_prediction','title','tag','label'],header=None,encoding='utf-8').astype(str)


# tmp = train_data.groupby('prefix')

tmp2 = train_data.sort_values(by='prefix')


tt = 1

