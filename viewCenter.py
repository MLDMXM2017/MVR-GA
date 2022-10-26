# -*-coding:utf-8 -*-

# File       : viewCenter.py
# Time       ：2022/4/16 22:32
# Author     ：BEE2E7
# Description：计算各视角中心点以及样本距离各视角中心点的贴近距离

scale_l=[0.15,0.25,0.1,0.03] #每个视角抽取特征的比例

global feature_index_l #各视角特征索引列表
global class1_center_l,class2_center_l,class12_distance_l #类别1中心列表、类别2中心列表、类别12中心之间距离
global feature_offset_l #各视角特征偏移量

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np


#计算样本距离视角view_id的中心点距离
def calculate_distance(view_id,sample_index,file):
    global feature_index_l
    global class1_center_l, class2_center_l, class12_distance_l
    global feature_offset_l

    #读物数据
    front_path = "dataSet/"
    if file==1:
        x_path=front_path+"x_valid.csv"  # 验证集x
        x_df=pd.read_csv(x_path)
    elif file==2:
        x_path=front_path+"x_test.csv" #测试集x
        x_df=pd.read_csv(x_path)

    x=x_df.iloc[[sample_index]]

    #获取本视角中心点相关信息
    f_index_l = feature_index_l[view_id] #特征索引列表（没有添加偏移量）
    class1_center=class1_center_l[view_id] #类别1中心位置
    class2_center=class2_center_l[view_id] #类别2中心位置
    class12_distance=class12_distance_l[view_id] #类别12中心之间距离

    #计算样本到两个中心点的距离
    distance1,distance2=0,0 #样本距离两个中心点的距离
    for i in range(len(f_index_l)):
        f_index=f_index_l[i]+feature_offset_l[view_id] #计算特征实际索引
        distance1 += (x[str(f_index)].values - class1_center[i]) ** 2
        distance2 += (x[str(f_index)].values - class2_center[i]) ** 2
    distance1 = distance1 ** 0.5
    distance2 = distance2 ** 0.5

    return distance1,distance2,class12_distance

#初始化 计算各视角中线点 结果计算在全局变量中
def init_center(view_amount,f_amount_l,f_offset_l):
    global feature_index_l
    global class1_center_l, class2_center_l, class12_distance_l
    global feature_offset_l

    feature_index_l=[]
    class1_center_l,class2_center_l,class12_distance_l=[],[],[]
    feature_offset_l=f_offset_l

    #遍历四个视角就行特征提取
    for v in range(view_amount):
        print("view "+str(v)+"center initial...")
        #读取数据
        front_path = "dataSet/"
        x_train_path = front_path + "x" + str(v) + "_train.csv"  # 训练集x
        x_valid_path = front_path + "x" + str(v) + "_valid.csv"  # 验证集x
        y_train_path = front_path + "y_train.csv"  # 训练集y
        y_valid_path = front_path + "y_valid.csv"  # 验证集y

        x_train_df = pd.read_csv(x_train_path)  # x训练集
        y_train_df = pd.read_csv(y_train_path).loc[:, 'Y']  # y训练集
        x_valid_df = pd.read_csv(x_valid_path)  # x验证集
        y_valid_df = pd.read_csv(y_valid_path).loc[:, 'Y']  # y验证集

        #将数据按有毒、无毒标签分开 class1有毒label=1;class2无毒label=0
        label_l = y_train_df.tolist()  # 标签列表
        class1_index = [i for i, item in enumerate(label_l) if item == 1]  # 标签为1
        class2_index = [i for i, item in enumerate(label_l) if item == 0]  # 标签为0

        # 有毒无毒的特征列
        class1_df = x_train_df.iloc[class1_index, :]  # 有毒特征列
        class2_df = x_train_df.iloc[class2_index, :]  # 无毒特征列
        class1_amount = class1_df.shape[0]  # 有毒样本数
        class2_amount = class2_df.shape[0]  # 无毒样本数

        importance_l, f_aver_impo = [], []  # 每次实验特征重要性列表、特征重要性平均值

        # 构建五次随机森林  特征重要性取均值啦
        for i in range(5):
            clf = RandomForestClassifier()  # 随机森林模型
            clf.fit(x_train_df, y_train_df)
            importance_l.append(clf.feature_importances_)

        # 计算特征重要性平均值
        for f_index in range(x_train_df.shape[1]):
            importance = (importance_l[0][f_index] + importance_l[1][f_index] + importance_l[2][f_index] +
                          importance_l[3][f_index] + importance_l[4][f_index]) / 5
            f_aver_impo.append(importance)

        sorted_id = sorted(range(len(f_aver_impo)), key=lambda k: f_aver_impo[k], reverse=True)  # 特征按重要性排序后的索引列表

        feature_num=int(f_amount_l[v]*scale_l[v]) #view_id筛选规则数目
        index_l = sorted_id[0:feature_num]
        feature_index_l.append(index_l) #视角v筛选出的特征索引列表

        #计算视角view_id的两个中心点，以及两个中线点之间的距离
        class1_center, class2_center = [], []  # 有毒无毒中心点的值
        class12_distance = 0

        for index in index_l:
            col1 = class1_df.loc[:, str(index + feature_offset_l[v])].tolist()
            col2 = class2_df.loc[:, str(index + feature_offset_l[v])].tolist()

            class1_point = (float)(sum(col1)) / class1_amount
            class2_point = (float)(sum(col2)) / class2_amount
            class1_center.append(class1_point)
            class2_center.append(class2_point)

            class12_distance += (class1_point - class2_point) ** 2

        class12_distance = class12_distance ** 0.5

        class1_center_l.append(class1_center) #类别1中心位置
        class2_center_l.append(class2_center) #类别2中心位置
        class12_distance_l.append(class12_distance) #类别12中心间距离

        print("class12_distance",class12_distance)
    return
