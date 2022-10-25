# -*- coding: utf-8 -*-
# @File    : beforeSelect.py
# @Author  : BEE2E7
# @Date    : 2022/4/18 
# @Desc    : 计算筛选前规则对样本预测情况

import csvRecord
import viewAcc

import pandas as pd

def acc_before_selct(view_amount,rule_amount_l):
    print("\ncalculate view acc before selecting...")

    # # 从文件中读取forest的acc
    # f_acc_path = "ruleInformation/forest_acc.csv"
    # f_acc_df = pd.read_csv(f_acc_path)
    # f_v_acc_l = f_acc_df.loc[:, 'f_v_acc'].tolist()
    # f_t_acc_l = f_acc_df.loc[:, 'f_t_acc'].tolist()

    #个体为规则全集
    ind_list=[]
    for amount in rule_amount_l:
        ind=[1]*amount
        ind_list.append(ind)

    v_acc_l, v_cover_l, v_error_l, v_evaluate_l, \
    t_acc_l, t_cover_l, t_error_l, t_evaluate_l = viewAcc.view_acc(ind_list)

    # 将结果记录在csv文件中
    # 创建文件
    file_l, write_l = csvRecord.create_file(0, "rule")
    rule_f = file_l[0]
    rule_w = write_l[0]

    # # 各视角规则情况记录
    csvRecord.set_index(" ")
    # csvRecord.record_line(rule_w, ["view_id", "f_v_acc", "f_t_acc", "r_v_acc", "r_t_acc", \
    #                                "", "r-f_v_acc", "r-f_t_acc", "r_v_cover", "r_t_cover"])
    # for v in range(view_amount):
    #     r_f_v_acc = (v_acc_l[v] - f_v_acc_l[v]) * 100
    #     r_f_t_acc = (t_acc_l[v] - f_t_acc_l[v]) * 100
    #
    #     data1 = [v, f_v_acc_l[v], f_t_acc_l[v], v_acc_l[v], t_acc_l[v]]
    #     data2 = [" ", r_f_v_acc, r_f_t_acc, v_cover_l[v], t_cover_l[v]]
    #     data = data1
    #     data.extend(data2)
    #
    #     csvRecord.record_line(rule_w, data)
    #
    # csvRecord.record_line(rule_w, [" "])
    csvRecord.record_line(rule_w, ["test"])

    print("\nrandom forest and rule situation recorded...")

    # 视角拼接情况记录

    # 测试集
    csvRecord.record_line(rule_w, ["V1", "V2", "V3", "V4", \
                                   "1+2", "1+3", "1+4", "2+3", "2+4", "3+4", \
                                   "1+2+3", "1+2+4", "1+3+4", "2+3+4", "1+2+3+4"])
    csvRecord.set_index("acc")
    csvRecord.record_line(rule_w, t_acc_l)
    csvRecord.set_index("cover")
    csvRecord.record_line(rule_w, t_cover_l)

    # 视角拼接测试集error
    csvRecord.set_index(" ")
    csvRecord.record_line(rule_w, ["error"])
    csvRecord.record_line(rule_w, ["c1", "c2", "e1", "e2", " ", \
                                   "accuracy", "f1_score", "auc", "recall_0", "recall_1", "precision_0", "precision_1"])

    index_list = ["v1", "v2", "v3", "v4", \
                  "1+2", "1+3", "1+4", "2+3", "2+4", "3+4", "1+2+3", "1+2+4", "1+3+4", "2+3+4", "1+2+3+4"]
    for i in range(len(index_list)):
        csvRecord.set_index(index_list[i])
        data = t_error_l[i]
        data.append(" ")
        data.extend(t_evaluate_l[i])
        csvRecord.record_line(rule_w, data)

    print("\nview concat on test recorded...")

    # 验证集
    csvRecord.set_index(" ")
    csvRecord.record_line(rule_w, [" "])
    csvRecord.record_line(rule_w, ["valid"])
    csvRecord.record_line(rule_w, ["V1", "V2", "V3", "V4", \
                                   "1+2", "1+3", "1+4", "2+3", "2+4", "3+4", \
                                   "1+2+3", "1+2+4", "1+3+4", "2+3+4", "1+2+3+4"])
    csvRecord.set_index("acc")
    csvRecord.record_line(rule_w, v_acc_l)
    csvRecord.set_index("cover")
    csvRecord.record_line(rule_w, v_cover_l)

    # 视角拼接测试集error
    csvRecord.set_index(" ")
    csvRecord.record_line(rule_w, ["error"])
    csvRecord.record_line(rule_w, ["c1", "c2", "e1", "e2", " ", \
                                   "accuracy", "f1_score", "auc", "recall_0", "recall_1", "precision_0", "precision_1"])

    index_list = ["v1", "v2", "v3", "v4", \
                  "1+2", "1+3", "1+4", "2+3", "2+4", "3+4", "1+2+3", "1+2+4", "1+3+4", "2+3+4", "1+2+3+4"]
    for i in range(len(index_list)):
        csvRecord.set_index(index_list[i])

        data = v_error_l[i]
        data.append(" ")
        data.extend(v_evaluate_l[i])
        csvRecord.record_line(rule_w, data)

    csvRecord.file_close(file_l)
    print("\nview concat on valid recorded...")

    return