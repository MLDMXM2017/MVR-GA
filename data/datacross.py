# -*- coding: utf-8 -*-
# @File    : datacross.py
# @Author  : BEE2E7
# @Date    : 2022/4/28 
# @Desc    :

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

#数据文件路径
fp_path="data/Version_6_fp_881.csv" #特征1——分子指纹_ECFP
expression_path="data/Version_6_expression_978.csv" #特征2——分子指纹_MACCS
target_path="data/Version_6_target_1023.csv" #特征3——分子指纹_Pubchem
phychem_path= "data/Version_6_phychem_200.csv"  #特征4——物化性质
label_path="data/Version_6_label_CID.csv" #标签

def cross():
    f1_df = pd.read_csv(fp_path)  # 特征1
    f2_df = pd.read_csv(expression_path)  # 特征2
    f3_df = pd.read_csv(target_path)  # 特征3
    f4_df = pd.read_csv(phychem_path)  # 特征4
    label_df = pd.read_csv(label_path)  # 标签

    # 合并特征 获取每个视角长度
    feature_df = pd.concat([f1_df, f2_df], axis=1)
    feature_df = pd.concat([feature_df, f3_df], axis=1)
    feature_df = pd.concat([feature_df, f4_df], axis=1)
    v1_col = f1_df.shape[1]
    v2_col = f2_df.shape[1]
    v3_col = f3_df.shape[1]
    v4_col = f4_df.shape[1]

    # 重新规划列名
    feature_columns = []
    for i in range(feature_df.shape[1]):
        feature_columns.append(str(i))
    feature_df.columns = feature_columns

    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0)
    x, y = np.array(feature_df), np.array(label_df)

    index=0

    for train_id, test_id in skf.split(x, y):
        print("index",index)
        x_train_l, x_test_l = x[train_id], x[test_id]
        y_train_l, y_test_l = y[train_id], y[test_id]

        x_d=pd.DataFrame(x_train_l)
        y_d=pd.DataFrame(y_train_l)
        x_test=pd.DataFrame(x_test_l)
        y_test=pd.DataFrame(y_test_l)

        y_d.columns = ['Y']
        y_test.columns = ['Y']

        x_train, x_valid, y_train, y_valid = train_test_split(x_d, y_d, test_size=0.25)  # 划分出验证集和测试集，shuffle=False不需要重新洗牌

        x1_train = x_train.iloc[:, 0:v1_col]  # view1训练集
        x2_train = x_train.iloc[:, v1_col:v1_col + v2_col]  # view2训练集
        x3_train = x_train.iloc[:, v1_col + v2_col:v1_col + v2_col + v3_col]  # view3训练集
        x4_train = x_train.iloc[:, v1_col + v2_col + v3_col:v1_col + v2_col + v3_col + v4_col]  # view4训练集

        x1_valid = x_valid.iloc[:, 0:v1_col]  # view1验证集
        x2_valid = x_valid.iloc[:, v1_col:v1_col + v2_col]  # view2验证集
        x3_valid = x_valid.iloc[:, v1_col + v2_col:v1_col + v2_col + v3_col]  # view3验证集
        x4_valid = x_valid.iloc[:, v1_col + v2_col + v3_col:v1_col + v2_col + v3_col + v4_col]  # view4验证集

        x1_test = x_test.iloc[:, 0:v1_col]  # view1测试集
        x2_test = x_test.iloc[:, v1_col:v1_col + v2_col]  # view2测试集
        x3_test = x_test.iloc[:, v1_col + v2_col:v1_col + v2_col + v3_col]  # view3测试集
        x4_test = x_test.iloc[:, v1_col + v2_col + v3_col:v1_col + v2_col + v3_col + v4_col]  # view4测试集

        front_path = "dataCrossResult" + str(index+1) + "/"

        # 保存数据
        x_train.to_csv(front_path + "x_train.csv", index=False, sep=',', encoding='utf_8')
        y_train.to_csv(front_path + "y_train.csv", index=False, sep=',', encoding='utf_8')
        x_valid.to_csv(front_path + "x_valid.csv", index=False, sep=',', encoding='utf_8')
        y_valid.to_csv(front_path + "y_valid.csv", index=False, sep=',', encoding='utf_8')
        x_test.to_csv(front_path + "x_test.csv", index=False, sep=',', encoding='utf_8')
        y_test.to_csv(front_path + "y_test.csv", index=False, sep=',', encoding='utf_8')

        x1_train.to_csv(front_path + "x0_train.csv", index=False, sep=',', encoding='utf_8')
        x2_train.to_csv(front_path + "x1_train.csv", index=False, sep=',', encoding='utf_8')
        x3_train.to_csv(front_path + "x2_train.csv", index=False, sep=',', encoding='utf_8')
        x4_train.to_csv(front_path + "x3_train.csv", index=False, sep=',', encoding='utf_8')

        x1_valid.to_csv(front_path + "x0_valid.csv", index=False, sep=',', encoding='utf_8')
        x2_valid.to_csv(front_path + "x1_valid.csv", index=False, sep=',', encoding='utf_8')
        x3_valid.to_csv(front_path + "x2_valid.csv", index=False, sep=',', encoding='utf_8')
        x4_valid.to_csv(front_path + "x3_valid.csv", index=False, sep=',', encoding='utf_8')

        x1_test.to_csv(front_path + "x0_test.csv", index=False, sep=',', encoding='utf_8')
        x2_test.to_csv(front_path + "x1_test.csv", index=False, sep=',', encoding='utf_8')
        x3_test.to_csv(front_path + "x2_test.csv", index=False, sep=',', encoding='utf_8')
        x4_test.to_csv(front_path + "x3_test.csv", index=False, sep=',', encoding='utf_8')

        index+=1
cross()
