# -*-coding:utf-8 -*-

# File       : csvRecord.py
# Time       ：2022/4/5 21:39
# Author     ：BEE2E7
# Description：结果记录文件

import csv

#规则提取部分记录文件
rule_name_l=["acc_before_ga.csv"]
rule_header_l=[]

#遗传算法部分记录文件
ga_name_l=["best_code.csv","fitness_best.csv","fitness_test.csv","fitness_pop.csv","fitness_evolution.csv"] #最优个体编码、最优个体适应度、测试情况、种群适应度、种群进化
ga_header_l=[["generation","fitness","ind_code"],["generation","fitness","accuracy","coverage"],["generation","fitness","accuracy","coverage"],\
            ["generation","fitness","accuracy","coverage"],["generation","min","max","avg","std"]] #文件表头

#最终结果记录文件
result_name_l=["acc_after_ga.csv","error_after_ga.csv"]
result_h1=["e_id","V1","V2","V3","V4",\
            "1+2", "1+3", "1+4", "2+3", "2+4", "3+4", \
            "1+2+3", "1+2+4", "1+3+4", "2+3+4", "1+2+3+4"," ", \
           "V1", "V2", "V3", "V4",  \
           "1+2", "1+3", "1+4", "2+3", "2+4", "3+4", \
           "1+2+3", "1+2+4", "1+3+4", "2+3+4", "1+2+3+4"]

result_h2=["view","c1","c2","e1","e2"]
result_header_l=[result_h1,result_h2]

global index,experiment_time

#设置expect_amount
def set_index(i):
    global index

    index=i

    return

#设置实验次数 （咋有种面向对象的感觉啊喂）
#传入:e_time 试验次数
#返回：空
def set_e_id(e_time):
    global experiment_time

    experiment_time=e_time

    return

#创建文件
def create_file(view_id,module):
    file_l,write_l=[],[]

    if module=="rule":
        name_l = rule_name_l
        header_l = rule_header_l
        front_path="result/"
    elif module=="ga":
        name_l = ga_name_l
        header_l = ga_header_l
        front_path ="result/view"+str(view_id)+"/"+str(experiment_time)+"_view"+str(view_id)+"_"
    elif module=="result":
        name_l = result_name_l
        header_l = result_header_l
        front_path ="result/"

    for i in range(len(name_l)):
        name=name_l[i]
        file_path=front_path+name
        if module=="rule" :
            f = open(file_path, 'a+', newline='')
        elif module=="ga":
            f=open(file_path,'w',newline='')
        elif module=="result":
            f = open(file_path, 'a+', newline='')

        write = csv.writer(f)
        if module=="ga" or module=="result":
            write.writerow(header_l[i])

        file_l.append(f)
        write_l.append(write)

    return file_l,write_l


#向文件写入一行信息
def record_line(write,data):
    global index

    row=[index]
    row.extend(data)

    write.writerow(row)

    return


#将缓存中的数据立即写入文件
def file_flush(file_list):
    for f in file_list:
        f.flush()

    return

#关闭文件
def file_close(file_list):
    for f in file_list:
        f.close()

    return
