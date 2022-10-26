# -*- coding: utf-8 -*-
# @File    : viewAcc.py
# @Author  : BEE2E7
# @Date    : 2022/4/18 
# @Desc    :
import viewSeparate
import viewConcat

def view_acc(ind_list):
    # 分别在单独和拼接情况下，计算各视角情况

    v_acc_l1, v_cover_l1, v_error_l1, v_evaluate_l1, \
    t_acc_l1, t_cover_l1, t_error_l1, t_evaluate_l1 = viewSeparate.seperate_acc(ind_list)  # 单视角

    v_acc_l2, v_cover_l2, v_error_l2, v_evaluate_l2, \
    t_acc_l2, t_cover_l2, t_error_l2, t_evaluate_l2 = viewConcat.concat_acc(ind_list)  # 视角拼接

    v_acc_l, v_cover_l, v_error_l, v_evaluate_l = v_acc_l1, v_cover_l1, v_error_l1, v_evaluate_l1
    t_acc_l, t_cover_l, t_error_l, t_evaluate_l = t_acc_l1, t_cover_l1, t_error_l1, t_evaluate_l1

    v_acc_l.extend(v_acc_l2)
    v_cover_l.extend(v_cover_l2)
    v_error_l.extend(v_error_l2)
    v_evaluate_l.extend(v_evaluate_l2)

    t_acc_l.extend(t_acc_l2)
    t_cover_l.extend(t_cover_l2)
    t_error_l.extend(t_error_l2)
    t_evaluate_l.extend(t_evaluate_l2)

    return v_acc_l, v_cover_l, v_error_l, v_evaluate_l, t_acc_l, t_cover_l, t_error_l, t_evaluate_l
