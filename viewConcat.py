# -*- coding: utf-8 -*-
# @File    : viewConcat.py
# @Author  : BEE2E7
# @Date    : 2022/4/18 
# @Desc    : 计算合并视角规则对样本的预测结果  和单视角预测模块结构上有很大相似之处



import viewCenter
import rules0
import rules1
import rules2
import rules3

from sklearn.metrics import classification_report,roc_auc_score
import pandas as pd

global x_valid, x_test, y_valid, y_test
global rule_amount_list, offset_list, view_amount, ind_length

#遍历规则集，对某一样本进行预测
def f(ind,sample_index,file):
    global x_valid,x_test
    global offset_list,view_amount

    #四视角结果统计
    view_toxics, view_non_toxics = [0, 0, 0, 0], [0, 0, 0, 0]  # 各视角有毒、无毒预测情况
    view_weights1, view_weights2 = [0, 0, 0, 0], [0, 0, 0, 0]  # 各视角预测使用规则权重情况

    # 从对应数据文件中取出样本
    if file == 1:
        x = x_valid.iloc[[sample_index]]
    elif file == 2:
        x = x_test.iloc[[sample_index]]

    #遍历view1中规则
    for rule_index in range(offset_list[0],offset_list[1]):
        if ind[rule_index]==1:
            rule_f = "rules0.rule" + str(rule_index)
            toxic_result = eval(rule_f)(x)

            if toxic_result==1: #当前规则认为有毒
                view_toxics[0]+=1

            elif toxic_result==0: #当前规则认为无毒
                view_non_toxics[0]+=1


    #视角2
    for rule_index in range(offset_list[1],offset_list[2]):
        if ind[rule_index]==1:
            rule_f = "rules1.rule" + str(rule_index-offset_list[1])
            toxic_result = eval(rule_f)(x)

            if toxic_result==1: #认为有毒
                view_toxics[1] += 1

            elif toxic_result==0: #认为无毒
                view_non_toxics[1] += 1


    #视角3
    for rule_index in range(offset_list[2],offset_list[3]):
        if ind[rule_index]==1:
            rule_f = "rules2.rule" + str(rule_index-offset_list[2])
            toxic_result = eval(rule_f)(x)

            if toxic_result == 1:  # 认为有毒
                view_toxics[2] += 1

            elif toxic_result == 0:  # 认为无毒
                view_non_toxics[2] += 1


    #视角4
    for rule_index in range(offset_list[3],offset_list[4]):
        if ind[rule_index]==1:
            rule_f = "rules3.rule" + str(rule_index-offset_list[3])
            toxic_result = eval(rule_f)(x)

            if toxic_result == 1:  # 认为有毒
                view_toxics[3] += 1

            elif toxic_result == 0:  # 认为无毒
                view_non_toxics[3] += 1


    #计算各视角预测结果 单个视角上 只信任非平局结果 对于平局结果不予考虑
    view_predicts = [-1, -1, -1, -1]  # 各视角的预测值
    toxic_list, non_toxic_list = [], []  # 认为有毒和无毒的视角列表
    dogfall_list=[] #平局视角
    for v in range(4):
        if view_toxics[v]>view_non_toxics[v]: #认为有毒>无毒
            view_predicts[v]=1
            toxic_list.append(v) #将视角id记入list
        elif view_toxics[v]<view_non_toxics[v]: #认为有毒<无毒
            view_predicts[v]=0
            non_toxic_list.append(v)
        elif view_toxics[v]!=0: #平局，但非不覆盖
            dogfall_list.append(v) #平局但非

    #综合四个视角
    sample_predict = -1  # 样本预测标签 样本真实标签
    dogfall=2 #平局情况（1 出现平局；2 不出现平局）

    if len(toxic_list)>len(non_toxic_list): #认为有毒视角>无毒
        sample_predict=1
    elif len(toxic_list)<len(non_toxic_list): #认为有毒视角<无毒
        sample_predict=0
    elif len(toxic_list)!=0 or len(dogfall_list)!=0: #出现平局 但覆盖
        dogfall=1 #出现平局

        distance_l, distance_label_l = [], []  # 距离数组、距离预测标签数组
        d_toxic_l, d_non_toxic_l = [], []  # 有毒、无毒标签统计

        for v_id in range(view_amount):
            d1, d2, d3 = viewCenter.calculate_distance(v_id, sample_index, file)  # 计算距离两个类别中心的距离

            distance_l.append(abs(d1 - d2) / d3)
            if d1 < d2:
                distance_label_l.append(1)
                d_toxic_l.append(v_id)
            elif d1 > d2:
                distance_label_l.append(0)
                d_non_toxic_l.append(v_id)

            if len(d_toxic_l) > len(d_non_toxic_l):  # 有毒次数>无毒次数
                sample_predict = 1
            elif len(d_toxic_l) < len(d_non_toxic_l):  # 有毒次数<无毒次数
                sample_predict = 0
            else:  # 又是杀千刀的平局
                d_label_index = distance_l.index(max(distance_l))
                sample_predict = distance_label_l[d_label_index]


    else: #len(toxic_list)=len(non_toxic_list)=0 且len(dog_fall_list)=0 即为四个视角均不覆盖 彻底没救了
        sample_predict=-1

    return sample_predict,dogfall


#计算多视角个体ind对数据集file中样本的预测情况
#传入：
#返回：
def calculate(ind,file):
    global x_valid,x_test,y_valid,y_test

    #选择数据集
    if file == 1:
        sample_amount = x_valid.shape[0]
    elif file == 2:
        sample_amount = x_test.shape[0]

    correct, error = 0, 0  # 当前个体对当前样本的预测情况
    c1, c2, e1, e2 = 0, 0, 0, 0  # 当前个体在当前个体上的错误情况
    ind_predict, ind_label = [], []  # 规则集预测结果、对应样本真实标签

    # 遍历所有样本，计算该个体在所有样本上的准确率和覆盖率
    for sample_index in range(sample_amount):
        sample_predict, dogfall = f(ind, sample_index,file)  #返回样本预测标签，以及是否出现平局（2 未出现；1 出现）

        if sample_predict==-1: #当前样本不被规则集覆盖
            continue

        # 获取样本真实标签
        if file == 1:
            sample_label = y_valid[sample_index]
        elif file == 2:
            sample_label = y_test[sample_index]

        ind_predict.append(sample_predict)  # 预测标签加入列表
        ind_label.append(sample_label)  # 真实标签加入列表

        if sample_predict==sample_label: #样本预测值和真实标签相同
            correct+=1
            if dogfall==1:
                c1+=1
            elif dogfall==2:
                c2+=1
        elif sample_predict!=sample_label: #样本预测值和真实标签不相同
            error+=1
            if dogfall==1:
                e1+=1
            elif dogfall==2:
                e2+=1

    report = classification_report(ind_label, ind_predict, output_dict=True)
    acc = report['accuracy']
    f1 = report['macro avg']['f1-score']
    auc = roc_auc_score(ind_label, ind_predict)
    recall_0 = report["0"]['recall']
    recall_1 = report["1"]['recall']
    precision_0 = report["0"]['precision']
    precision_1 = report["1"]['precision']
    evaluate_list = [acc, f1, auc, recall_0, recall_1, precision_0, precision_1]

    coverage = float(len(ind_predict)) / sample_amount  # 计算覆盖率

    print("\ncorrect: ", correct, "  error: ", error)
    print("矛盾正确", c1, " 不矛盾正确", c2, " 矛盾错误", e1, " 不矛盾错误", e2)

    return acc, coverage, c1, c2, e1, e2, evaluate_list

#初始化
def initial(r_amount_l,r_offset_l):
    global x_valid,x_test,y_valid,y_test
    global rule_amount_list,offset_list,  view_amount,ind_length

    # 初始化全局变量
    x_valid_file = "dataSet/x_valid.csv"  # 特征验证集文件位置
    y_valid_file = "dataSet/y_valid.csv"  # 标签验证集文件位置
    x_test_file = "dataSet/x_test.csv"  # 特征测试集文件位置
    y_test_file = "dataSet/y_test.csv"  # 表现测试集文件位置

    # 初始化全局变量
    x_valid = pd.read_csv(x_valid_file)  # 特征验证集
    y_valid = pd.read_csv(y_valid_file).loc[:, 'Y']  # 标签验证集
    x_test = pd.read_csv(x_test_file)  # 特征测试集
    y_test = pd.read_csv(y_test_file).loc[:, 'Y']  # 标签测试集

    rule_amount_list=r_amount_l
    offset_list = r_offset_l  # 视角规则偏移量

    view_amount = len(rule_amount_list)
    ind_length = sum(rule_amount_list)  # 个体长度

    return

def concat_acc(ind_list):
    global rule_amount_list

    concat_list = ["1+2", "1+3", "1+4", "2+3", "2+4", "3+4", \
                   "1+2+3", "1+2+4", "1+3+4", "2+3+4", "1+2+3+4"]

    v_acc_l,v_cover_l,v_error_l=[],[],[]
    t_acc_l,t_cover_l,t_error_l=[],[],[]
    evaluate_v,evaluate_t=[],[]

    for c in concat_list:
        v_list=[0,0,0,0] #视角列表

        c_list=c.split('+')
        for x in c_list:
            v=int(x)
            v_list[v-1]=1


        ind=[]
        for v in range(view_amount):
            if v_list[v]==1:
                ind.extend(ind_list[v])
            else:
                sub_ind=[0]*rule_amount_list[v]
                ind.extend(sub_ind)

        print("\nview concat", c)
        v_acc, v_cover, v_c1, v_c2, v_e1, v_e2, evaluate_l1 = calculate(ind, 1)
        print("验证集  acc", v_acc, " cover1", v_cover, " c1", v_c1, " c2", v_c2, " e1", v_e1, " e2", v_e2)
        t_acc, t_cover, t_c1, t_c2, t_e1, t_e2, evaluate_l2 = calculate(ind, 2)
        print("测试集 acc", t_acc, "cover2", t_cover, " c1", t_c1, " c2", t_c2, " e1", t_e1, " e2", t_e2)

        v_acc_l.append(v_acc)
        v_cover_l.append(v_cover)
        v_error_l.append([v_c1, v_c2, v_e1, v_e2])
        evaluate_v.append(evaluate_l1)

        t_acc_l.append(t_acc)
        t_cover_l.append(t_cover)
        t_error_l.append([t_c1, t_c2, t_e1, t_e2])
        evaluate_t.append(evaluate_l2)

    print("\n")

    return v_acc_l, v_cover_l, v_error_l, evaluate_v, t_acc_l, t_cover_l, t_error_l, evaluate_t

