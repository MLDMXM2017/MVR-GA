# -*-coding:utf-8 -*-

# File       : getRule.py
# Time       ：2022/4/5 18:23
# Author     ：BEE2E7
# Description：

import pandas as pd
import numpy as np
import datetime

global rule_features,rule_thresholds,rule_symbols,rule_values,rule_sample,rule_predict,rule_index_t,sorted_features #规则特征 规则阈值 规则符号 规则返回值 规则样本数目 规则预测值 规则索引表 规则特征排序表
global rule_amount,rule_pool_size,all_rule_amount,same_rule_amount,useless_rule_amount,sorted_length,gini_zero_amount #当前规则数目 规则池大小 供选择规则总数目 重复规则数目 作废规则数目 规则排序特征表长度 gini=0规则数目
global samples_not_enough

RULE_SAMPLE=3
#删除在索引表中位置为pos的规则，主要是处理特征排序表和规则索引表，因为规则内容表中的内容会被新内容替换
#传入：pos 规则在索引表中的位置
#返回：空
def rule_delete(pos):
    global rule_index_t


    sorted_id = rule_index_t[pos][2]  # 获取特征在排序列表中的位置
    sorted_index=(int)(sorted_id-1) #获取排序特征id

    index_list = [i for i, item in enumerate(rule_index_t) if item[2] == sorted_id]  # 查找使用相同特征的规则在索引表中的位置
    #根据规则特征出现次数，采取不同的策略删除特征排序列表中的内容
    if len(index_list)==1: #该组特征只在规则池中出现一次，则可以直接删除
        sorted_index=int(sorted_index)
        sorted_features[sorted_index]=[-1,-1]
    else: #该组特征在规则池中多次出现，不能直接删除，需修改索引表中所有对应表项数值

        for index in index_list:

            rule_index_t[index][3] = rule_index_t[index][3] - 1

    return

#当规则特征重复的情况下，查看规则池中是否有重复规则
#传入：path_features 规则特征、path_thresholds规则阈值、path_symbols规则符号、sorted_index规则特征在排序列表中的索引
#返回：is_same 规则是否相同、times当前特征在规则池中出现次数
def rule_equal(path_features,path_thresholds,path_symbols,sorted_index):

    global rule_features,rule_thresholds,rule_symbols,rule_index_t,sorted_features

    is_same=True
    sorted_f=sorted_features[sorted_index] #特征排序后序列
    sorted_id=sorted_index+1 #排序特征id

    index_list = [i for i, item in enumerate(rule_index_t) if item[2] == sorted_id]  # 查找使用相同特征的规则在索引表中的位置


    #遍历所有使用相同特征的规则，查看规则内容是否相同
    for i in range(len(index_list)):
        is_same=True
        index=index_list[i]

        pos = int(rule_index_t[index][1])  # 获得规则在规则内容列表的位置
        other_features,other_thresholds,other_symbols,other_predict=rule_features[pos],rule_thresholds[pos],rule_symbols[pos],rule_predict[pos] #获取规则内容

        for j in range(len(sorted_f)): #逐特征比较规则内容
            f=sorted_f[j]
            x1=path_features.index(f)
            x2=other_features.index(f)
            if path_thresholds[x1]!=other_thresholds[x2] or path_symbols[x1]!=other_symbols[x2]:
                is_same=False
                break
        if is_same==True: #规则重复

            break

    if is_same==False: #规则特征重复，但规则内容不重复，修改索引表中规则特征出现次数
        for index in index_list:

            rule_index_t[index][3] += 1


    return is_same, rule_index_t[index_list[0]][3]  # 规则是否相同、规则特征重复次数

#将规则特征按序号降序进行排序
#传入：规则特征列表
#返回：排序后的规则特征列表
def feature_sort(path_features):
    sorted_features=[]

    for i in range(len(path_features)):
        max_f=max(path_features)
        max_index=path_features.index(max_f)
        sorted_features.append(max_f)
        path_features[max_index]=-1

    return sorted_features

#获取规则内容
#传入：tree 决策树、path 规则路径
#返回：path_features 规则特征、path_thresholds 特征阈值、path_symbols 规则符号、path_predict 规则返回值、path_sample 规则样本
def get_rule_content(tree,path):
    #从决策树中获取所需结构
    children_left=tree.tree_.children_left #节点x的左孩子，如果x是叶子节点，则左孩子为-1
    children_right=tree.tree_.children_right #节点x的右孩子，如果x是叶子节点，则右孩子为-1
    features=tree.tree_.feature #节点所用特征
    thresholds=tree.tree_.threshold #节点特征阈值
    values=tree.tree_.value #节点的值
    samples=tree.tree_.n_node_samples #节点的样本数目

    path_features,path_thresholds,path_symbols,path_values=[],[],[],[] #当前路径对应内容
    path_predict=0

    #遍历路径中的节点，但不包括叶子节点
    for index in range(len(path)-1):
        node_index=path[index] #获取当前节点在决策树中的索引
        next_index=path[index+1] #获取下一个节点的索引
        if children_left[node_index]==next_index: #下一个节点是当前节点的左孩子，即当前特征值判定符号为<=，用1表示
            path_symbols.append(1)
        elif children_right[node_index]==next_index: #下一个节点是当前节点的右孩子，即当前特征值判定符号为>，用2表示
            path_symbols.append(2)

        path_features.append(features[node_index])
        path_thresholds.append(thresholds[node_index])

    #根据叶子节点判断规则输出值
    index+=1
    node_index=path[index]
    value1,value2=values[node_index][0][0],values[node_index][0][1]
    if value1>value2: #规则返回值为0
        path_predict=0
    elif value1<value2: #规则返回值为1
        path_predict=1
    else: #value1等于value2 即gini=0.5 返回-1 此规则作废
        path_predict=-1

    path_values.append(value1)
    path_values.append(value2)

    path_sample=samples[node_index]
    return path_features,path_thresholds,path_symbols,path_values,path_sample,path_predict


#当前规则与规则池中规则进行比较，判断是否将当前规则加入规则池
#传入：k 规则gini系数/样本函数、tree 决策树、path规则路径
#返回：空
def rules_compare(k, path_features, path_thresholds, path_symbols,path_values,path_predict, path_sample):

    global rule_features,rule_thresholds,rule_symbols,rule_predict,rule_index_t,sorted_features,rule_sample,rule_values
    global rule_amount,rule_pool_size,same_rule_amount,useless_rule_amount,sorted_length


    if path_predict==-1: #规则作废
        useless_rule_amount+=1
        return

    if rule_amount<rule_pool_size: #规则池未满
        # 规则池未满，查找是否有相同规则
        sorted_f = feature_sort(path_features.copy())  # 将规则特征按从大到小的顺序进行排序
        if sorted_f in sorted_features:  # 规则所使用特征重复，需判断规则整体是否重复

            sorted_index = sorted_features.index(sorted_f)  # 获得特征有序序列在表中的位置

            is_same, times = rule_equal(path_features, path_thresholds, path_symbols,  sorted_index)
            if is_same == True:  # 规则重复
                same_rule_amount += 1
                return
            else:  # 规则特征相同，但规则不重复
                rule_index_t.append([k, rule_amount, sorted_index + 1, times])
                rule_features.append(path_features)  # 规则特征
                rule_thresholds.append(path_thresholds)  # 规则阈值
                rule_symbols.append(path_symbols)  # 规则符号
                rule_values.append(path_values) #规则返回值
                rule_predict.append(path_predict)  # 规则返回值
                rule_sample.append(path_sample)
                rule_amount += 1

        else:  # 规则特征不重复
            rule_index_t.append([k, rule_amount, sorted_length + 1, 1])
            rule_features.append(path_features)  # 规则特征
            rule_thresholds.append(path_thresholds)  # 规则阈值
            rule_symbols.append(path_symbols)  # 规则符号
            rule_values.append(path_values)  # 规则返回值
            rule_predict.append(path_predict)  # 规则返回值
            rule_sample.append(path_sample) #规则样本数
            sorted_features.append(sorted_f)  # 规则特征排序序列
            rule_amount += 1  # 当前规则数目+1
            sorted_length += 1  # 当前规则特征排序序列+1

    else: #规则池满了
        pos=rule_amount-1

        rule_index_t=np.array(rule_index_t)
        rule_index_t = rule_index_t[np.lexsort(-rule_index_t[:, ::-1].T)]  # 按照表项第一个值gini/samples升序排序


        worst_k, worst_index, worst_sorted_index, worst_times = rule_index_t[pos][0], rule_index_t[pos][1], \
                                                                rule_index_t[pos][2], rule_index_t[pos][3]  # 获取最差规则的索引表项

        if k>worst_k: #当前规则有资格进行替换

            sorted_f = feature_sort(path_features.copy())  # 将规则特征按从大到小的顺序进行排序

            if sorted_f in sorted_features:  # 规则所使用特征重复，需判断规则整体是否重复

                sorted_index = sorted_features.index(sorted_f)  # 获得特征有序序列在表中的位置
                is_same, times = rule_equal(path_features, path_thresholds, path_symbols, sorted_index) #判断规则内容是否重复，若规则内容重复，则修改规则索引表
                if is_same==True: #规则内容重复，无需进行替换
                    same_rule_amount+=1
                    return
                else: #规则特征重复 但规则内容未重复，需要进行替换

                    rule_delete(pos) #删除索引表中位置为pos的规则
                    rule_index_t[pos][0], rule_index_t[pos][2], rule_index_t[pos][
                        3] = k, sorted_index + 1, times  # 用新规则的信息替换淘汰规则的信息 ginni/samples 特征排序位置 特征重复次数
                    rule_index = int(rule_index_t[pos][1])  # 获取被淘汰规则在内容列表中的位置
                    rule_features[rule_index]=path_features #替换规则特征
                    rule_thresholds[rule_index]=path_thresholds #替换特征阈值
                    rule_symbols[rule_index]=path_symbols #替换特征符号
                    rule_values[rule_index]=path_values
                    rule_predict[rule_index]=path_predict #替换规则返回值
                    rule_sample[rule_index]=path_sample
            else: #规则使用的特征组从未在规则池中出现

                rule_delete(pos) #删除被淘汰规则
                rule_index_t[pos][0], rule_index_t[pos][2], rule_index_t[pos][3] = k, sorted_length + 1, 1
                rule_index = int(rule_index_t[pos][1])  # 获取淘汰规则在规则内容列表中的位置
                rule_features[rule_index]=path_features #替换规则特征
                rule_thresholds[rule_index]=path_thresholds #替换规则阈值
                rule_symbols[rule_index]=path_symbols #替换规则符号
                rule_predict[rule_index]=path_predict #替换规则预测值
                rule_sample[rule_index]=path_sample #替换规则样本数目
                rule_values[rule_index]=path_values #替换规则返回值
                sorted_features.append(sorted_f) #写入排序特征
                sorted_length+=1 #排序规则数目+1

    return


#递归获得当前决策树中的路径 并将获得的规则传入规则比较函数
#传入：tree决策树、path 路径、length 当前路径长度、node_index 当前节点在树中的索引
#返回：空
def get_tree_paths(tree,path,length,node_index):
    global all_rule_amount,gini_zero_amount #所有规则数目 作废规则数目
    global samples_not_enough #gini=0 sample<11的规则数目

    children_left=tree.tree_.children_left[node_index] #节点node_index的左孩子，若左孩子为-1，则当前节点为叶子节点
    children_right=tree.tree_.children_right[node_index] #节点node_index的右孩子，若右孩子为-1，则当前节点为叶子节点
    impurity=tree.tree_.impurity[node_index] #节点node_index的gini系数

    path.append(node_index) #将当前节点压入路径path中

    if children_left==-1: #node_index为叶子节点，即当前规则已经遍历到尽头
        path_features, path_thresholds, path_symbols, path_values, path_sample, path_predict = get_rule_content(tree,path)

        if impurity==0:
            gini_zero_amount+=1
            if path_sample<RULE_SAMPLE:
                samples_not_enough+=1

        all_rule_amount+=1 #规则总数+1
        if path_sample>=RULE_SAMPLE:
            k = (float)(abs(path_values[0] - path_values[1])) / path_sample
            rules_compare(k, path_features, path_thresholds, path_symbols,path_values,path_predict, path_sample)  # 将当前规则与规则池中的规则进行比较
    else: #当前节点为非叶子节点
        get_tree_paths(tree,path,length+1,children_left)
        get_tree_paths(tree,path,length+1,children_right)

    path.pop(length) #将当前节点
    return

#将规则以函数形式写入py文件中
#传入：空 （因为数据都在全局变量中）
#返回：空
def write_rules(feature_offset,view_id):
    global rule_features,rule_thresholds,rule_symbols,rule_predict

    py_f=open("rules"+str(view_id)+".py","w")
    for rule_index in range(rule_amount): #遍历所有规则
        # py_f.write("def rule"+str(rule_offset+rule_index)+"(X):\n") #函数名
        py_f.write("def rule" + str( rule_index) + "(X):\n")  # 函数名
        features,thresholds,symbols,predict=rule_features[rule_index],rule_thresholds[rule_index],rule_symbols[rule_index],rule_predict[rule_index] #规则内容

        for index in range(len(features)):
            py_f.write("{space}"
                       "if X['{feature}'].values"
                .format(
                space=(index + 1) * "    ",
                feature=features[index]+feature_offset))
            if symbols[index] == 1:
                py_f.write(" <= ")
            else:
                py_f.write(" > ")

            py_f.write("{threshold} :"
                       "\n".format(
                threshold=thresholds[index]))
        index += 1
        py_f.write("{space}"
                   "return {v}"
                   "\n".format(
            space=(index + 1) * "    ",
            v=predict))

        while index > 0:
            py_f.write("{space}"
                       "else:"
                       "\n"
                       "{space2}"
                       "return -1"
                       "\n".format(
                space=(index) * "    ",
                space2=(index + 1) * "    "))
            index -= 1

    py_f.close()

    return

# 从随机森林中抽取规则
# 传入: forest 随机森林、forest_size 森林规模、expect_amount 期望规则数目、view_id 视角编号、depth 深度、experiment_id 实验编号、f_offset特征偏移量、r_offset规则偏移量
# 返回:
def get_rule(forest,forest_size,expect_amount,view_id,f_offset):
    global rule_features, rule_thresholds, rule_symbols, rule_values, rule_sample, rule_predict, rule_index_t, sorted_features  # 规则特征 规则阈值 规则符号 规则返回值 规则样本数目 规则预测值 规则索引表 规则特征排序表
    global rule_amount, rule_pool_size, all_rule_amount, same_rule_amount, useless_rule_amount, sorted_length, gini_zero_amount  # 当前规则数目 规则池大小 供选择规则总数目 重复规则数目 作废规则数目 规则排序特征表长度 gini=0规则数目
    global samples_not_enough  # gini=0 但samples不满足要求的规则

    rule_features, rule_thresholds, rule_symbols, rule_values, rule_sample, rule_predict, sorted_features = [], [], [], [], [], [], []
    rule_amount, all_rule_amount, same_rule_amount, useless_rule_amount, sorted_length, gini_zero_amount = 0, 0, 0, 0, 0, 0
    rule_pool_size = expect_amount  # 规则池大小=期望的规则数目

    rule_index_t=[]  # 规则索引表，依次存放：规则权重abs(value1-value2)/sample、规则在各表中位置rule_index、规则在特征排序表中位置sorted_index、规则排序特征出现次数times

    samples_not_enough = 0

    for tree_index in range(forest_size):
        path = []
        get_tree_paths(forest[tree_index], path, 0, 0)  # 获取树的路径

    #将规则内容保存到csv文件中，以便后续查看
    features_df=pd.DataFrame(rule_features) #规则特征list转DataFrame
    thresholds_df=pd.DataFrame(rule_thresholds) #规则阈值
    symbols_df=pd.DataFrame(rule_symbols) #规则符号
    values_df=pd.DataFrame(rule_values) #规则返回值

    r_max_len=features_df.shape[1] #阈值df的列数即为规则最大长度
    f_col,t_col,s_col=[],[],[]
    for i in range(r_max_len):
        f_col.append("fea_"+str(i+1))
        t_col.append("thre_"+str(i+1))
        s_col.append("sym_"+str(i+1))
    v_col=["v1","v2"]
    features_df.columns=f_col
    thresholds_df.columns=t_col
    symbols_df.columns=s_col
    values_df.columns=v_col

    rule_len=[]
    for feature in rule_features:
        rule_len.append(len(feature))



    df=pd.concat([features_df,thresholds_df],axis=1)
    df=pd.concat([df,symbols_df],axis=1)
    df=pd.concat([df,values_df],axis=1)
    df['sample']=rule_sample #规则样本数目
    df['predict']=rule_predict #规则预测值
    df['len']=rule_len #规则长度 用于最后可解释性

    front_path1="ruleInformation/v"+str(view_id)+"_"
    df.to_csv(front_path1+"rule.csv",index=False,sep=',',encoding="utf_8") #保存到csv文件中

    df2=pd.DataFrame(rule_index_t)
    front_path2="ruleIndex/v"+str(view_id)+"_"
    df2.to_csv(front_path2+"rule_index_table.csv",index=False,sep=',',encoding='utf_8')

    #将规则以函数形式保存在python文件中
    write_rules(f_offset,view_id)
    print("expect_amount ",expect_amount)
    print("all_rule_amount ", all_rule_amount)
    print("rule_amount", rule_amount)
    print("same_rule_amount", same_rule_amount)
    print("useless_rule_amount", useless_rule_amount)
    print("gini_zero", gini_zero_amount)
    print("samples not enough ", samples_not_enough)
    print("\n")

    #存一个txt文件
    # rule_amount 筛选出的规则数目、all_rule_amount 总规则数目、gini_zero_amount gini=0规则、
    # sample_not_enough gini=0但样本数小于阈值的股则、same_rule_amount相同规则、useless_rule_amount 作废规则
    now_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(now_time, '%Y-%m-%d %H:%M:%S')
    with open("ruleInformation/ruleAmount.txt", "a+") as f:
        f.write(time_str+"view"+str(view_id)+"\n")
        f.write("expect_amount"+str(expect_amount)+"\n")
        f.write("all_rule_amount "+str(all_rule_amount)+"\n")
        f.write("rule_amount"+str(rule_amount)+"\n")
        f.write("same_rule_amount"+str(same_rule_amount)+"\n")
        f.write("useless_rule_amount"+str(useless_rule_amount)+"\n")
        f.write("gini_zero"+str(gini_zero_amount)+"\n")
        f.write("samples not enough "+str(samples_not_enough)+"\n")
    f.close()

    return rule_amount