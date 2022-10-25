# -*- coding: utf-8 -*-
# @File    : mainGa.py
# @Author  : BEE2E7
# @Date    : 2022/4/18 
# @Desc    : 规则筛选部分主函数

evolution_n=50 #进化代数
pop_size=50 #种群大小

rule_amount1=500
rule_amount2=700
rule_amount3=500
rule_amount4=1000
# rule_amount1=750
# rule_amount2=1000
# rule_amount3=750
# rule_amount4=1000

# # #测试数据
# evolution_n=2 #进化代数
# pop_size=5 #种群大小
#
# rule_amount1=50
# rule_amount2=50
# rule_amount3=50
# rule_amount4=50

view_l=[0,1,2,3] #存放需要进化的视角
# view_l=[1,2] #存放需要进化的视角
view_amount=4
feature_amount_l=[881,978,1023,200] #各视角特征数目
f_offset_l=[0,881,1859,2882] #特征偏移量

rule_amount_list=[rule_amount1,rule_amount2,rule_amount3,rule_amount4] #视角规则数目列表
r_offset_list=[0,rule_amount1,rule_amount1+rule_amount2,rule_amount1+rule_amount2+rule_amount3,sum(rule_amount_list)] #规则偏移量列表

experiment_amount=1 #进化次数

#引入其他模块
# import ruleWeight #规则权重  不要了好吧
import viewCenter #中心距离
import viewSeparate #单视角预测
import viewConcat  #合并视角预测
import beforeSelect  #规则全集
import csvRecord  #记录文件
import ga  #遗传算法
import viewAcc  #不知道是个啥
import pandas as pd

#模块初始化

viewCenter.init_center(view_amount,feature_amount_l,f_offset_l)
viewSeparate.initial(view_amount)
viewConcat.initial(rule_amount_list,r_offset_list)

#计算进化前视角情况
beforeSelect.acc_before_selct(view_amount,rule_amount_list)

#创建记录文件
file_l,write_l=csvRecord.create_file(0,"result")
acc_f,acc_w=file_l[0],write_l[0]
error_f,error_w=file_l[1],write_l[1]

#重复3次进化
for e in range(experiment_amount):
    csvRecord.set_e_id(e)  # 设置实验id


    for v in view_l:
        print("\n\n@@@@@@@@@@@@---e", e, "view", v, "---@@@@@@@@@@@@")
        best_ind=ga.genetic_algorithm(rule_amount_list[v], evolution_n, pop_size, v)
        # best_ind_list.append(best_ind)

    #从文件中读取四个best_ind
    best_ind_list = []
    for v in range(view_amount):
        front_path ="result/view"+str(v)+"/"+str(e)+"_view"+str(v)+"_"
        file_path=front_path+"best_code.csv"

        code_l=pd.read_csv(file_path).loc[:, 'ind_code'].tolist() #个体编码列
        code_str=code_l[evolution_n] #最后一代个体 数据类型为str
        code_str=code_str[1:len(code_str)-1] #切片去除中括号

        code_str_l=code_str.split(',') #切分成单个数字
        ind=[]
        for st in code_str_l: #str类型转数值类型
            if st=='1' or st==' 1':
                ind.append(1)
            elif st=='0' or st==' 0':
                ind.append(0)

        best_ind_list.append(ind)


    concat_v_acc_l, concat_v_cover_l, concat_v_error_l, concat_v_evaluate_l,\
    concat_t_acc_l, concat_t_cover_l, concat_t_error_l, concat_t_evaluate_l = viewAcc.view_acc(best_ind_list)

    # 将结果记录在文件中
    csvRecord.set_index(e)

    # 测试集acc和cover
    data = concat_t_acc_l
    data.append(" ")
    data.extend(concat_t_cover_l)
    csvRecord.record_line(acc_w, data)

    # 测试集error结果
    csvRecord.record_line(error_w, " ")
    index_list = ["v1", "v2", "v3", "v4", \
                  "1+2", "1+3", "1+4", "2+3", "2+4", "3+4", "1+2+3", "1+2+4", "1+3+4", "2+3+4", "1+2+3+4"]

    for i in range(len(index_list)):
        csvRecord.set_index(index_list[i])

        data = concat_t_error_l[i]
        data.append(" ")
        data.extend(concat_t_evaluate_l[i])
        csvRecord.record_line(error_w, data)

    # 验证集情况
    csvRecord.set_index(" ")
    csvRecord.record_line(error_w, ["valid"])
    header = ["V1", "V2", "V3", "V4", \
              "1+2", "1+3", "1+4", "2+3", "2+4", "3+4", \
              "1+2+3", "1+2+4", "1+3+4", "2+3+4", "1+2+3+4", " ", \
              "V1", "V2", "V3", "V4", \
              "1+2", "1+3", "1+4", "2+3", "2+4", "3+4", \
              "1+2+3", "1+2+4", "1+3+4", "2+3+4", "1+2+3+4"]

    csvRecord.record_line(error_w, header)
    data = concat_v_acc_l
    data.append(" ")
    data.extend(concat_v_cover_l)
    csvRecord.record_line(error_w, data)

    csvRecord.record_line(error_w, " ")

    index_list = ["v1", "v2", "v3", "v4", \
                  "1+2", "1+3", "1+4", "2+3", "2+4", "3+4", "1+2+3", "1+2+4", "1+3+4", "2+3+4", "1+2+3+4"]
    for i in range(len(index_list)):
        csvRecord.set_index(index_list[i])

        data = concat_v_error_l[i]
        data.append(" ")
        data.extend(concat_v_evaluate_l[i])
        csvRecord.record_line(error_w, data)
    csvRecord.set_index(" ")
    csvRecord.record_line(error_w, " ")

    csvRecord.file_flush(file_l)


