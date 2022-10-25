# -*- coding: utf-8 -*-
# @File    : mainRule.py
# @Author  : BEE2E7
# @Date    : 2022/4/18 
# @Desc    : 规则抽取部分主函数

FOREST_SIZE=[100,100,100,100] #四个视角随机森林规模
TREE_DEPTH=[7,4,15,10] #四个视角随机森林的最大深度
RULE_POOL=[500,700,500,1000] #规则池大小
# RULE_POOL=[750,1000,750,1000] #规则池大小
# #测试数据
# FOREST_SIZE=[10,10,10,10] #四个视角随机森林规模
# TREE_DEPTH=[10,10,10,10] #四个视角随机森林的最大深度
# RULE_POOL=[50,50,50,50,] #规则池大小


import forestBuild #随机森林模型构建
import getRule #抽取规则

import pandas as pd
import datetime

f_offset_l=[0,881,1859,2882] #特征偏移量
view_l=[0,1,2,3] #视角列表 通过遍历视角列表来生成规则  改变列表就能重新生成规则
view_l=[1] #视角列表 通过遍历视角列表来生成规则  改变列表就能重新生成规则


#构建随机森林
forest_l,f_v_acc_l,f_t_acc_l=[],[],[]


for v in view_l: #遍历视角列表生成规则

    print("\n\n------view_",v,"------")  #视角生成规则
    #生成对应视图的随机森林  只需要返回森林  森林准确率记录在视角对应txt文件中  四个视角独立生成随机森林  互不干扰
    forest, v_acc, t_acc = forestBuild.build_forest(FOREST_SIZE[v], TREE_DEPTH[v], v) #创建随机森林

    #生成规则   输出acc和规则数目   并将acc和数目保存在txt文件中
    amount = getRule.get_rule(forest, FOREST_SIZE[v], RULE_POOL[v], v, f_offset_l[v])


    #输出acc和数目
    print(" valid_acc=",v_acc,"  test_acc=",t_acc)
    print("final_rule_amount",amount)

    #将情况存入txt文件中
    now_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(now_time, '%Y-%m-%d %H:%M:%S')
    with open("ruleInformation/view"+str(v)+".txt", "a+") as f:
        f.write(time_str + "\n")
        f.write("\n"+"valid_acc:" + str(v_acc) + " test_acc:" + str(t_acc) + "\n")
        f.write("rule_amount:" + str(amount) + "\n\n\n")
    f.close()

