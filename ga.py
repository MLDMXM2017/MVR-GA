# -*- coding: utf-8 -*-
# @File    : ga.py
# @Author  : BEE2E7
# @Date    : 2022/4/18
# @Desc    : 遗传算法模块

import csvRecord
import viewSeparate


import random
from deap import creator, base, tools, algorithms
from scipy.stats import bernoulli
import numpy as np

global view_id #视角编号
global best_code_file, fitness_best_file, fitness_test_file, fitness_pop_file, fitness_evolution_file  # 文件
global best_code_write, fitness_best_write, fitness_test_write, fitness_pop_write, fitness_evolution_write  # 写入
global best_ind, best_accuracy, best_coverage,best_fitness, best_c1, best_c2, best_e1, best_e2  # 最优个体编码、准确率、覆盖率、正确1、正确2、错误1、错误2


# 计算种群适应度 并将情况记录在文件中
# 传入：
# 返回：
def evaluate(population,generation):
    global view_id
    global best_ind, best_accuracy, best_coverage, best_fitness,best_c1, best_c2, best_e1, best_e2  # 最优个体编码、准确率、覆盖率、正确1、正确2、错误1、错误2
    global best_code_write, fitness_best_write,  fitness_pop_write  # 写入

    fitness_list=np.zeros(len(population)) #适应度列表
    acc_cover_list=[] #错误情况记录

    #遍历种群计算适应度
    for ind_index in range(len(population)):
        accuracy,coverage,c1,c2,e1,e2,eval_l=viewSeparate.calculate(view_id,population[ind_index],1)
        fitness_list[ind_index] = accuracy * coverage
        print("个体 ", ind_index, "accuracy ", accuracy, "coverage ", coverage, " fitness",
              accuracy * coverage,"\n" )

        csvRecord.record_line(fitness_pop_write,[accuracy * coverage,accuracy,coverage,c1,c2,e1,e2])

        acc_cover_list.append([accuracy,coverage,c1,c2,e1,e2])

    print(fitness_list) #输出适应度情况
    max_fitness=max(fitness_list) #找出本次计算中出现的最大适应度

    #查看是否出现了更优个体，若出现，则进行更新
    if max_fitness>best_fitness:
        best_fitness=max_fitness

        fitness_l = fitness_list.tolist()
        max_index = fitness_l.index(best_fitness)

        best_ind = population[max_index]
        best_acc_cover=acc_cover_list[max_index]
        best_accuracy,best_coverage=best_acc_cover[0],best_acc_cover[1]
        best_c1,best_c2,best_e1,best_e2=best_acc_cover[2],best_acc_cover[3],best_acc_cover[4],best_acc_cover[5]

    csvRecord.record_line(fitness_best_write,[best_fitness,best_accuracy,best_coverage,c1,c2,e1,e2])
    csvRecord.record_line(best_code_write,[best_fitness,best_ind])

    return fitness_list

#遗传算法寻优 并将进化过程记录在文件中
#传入：
#返回：最优个体
def genetic_algorithm(ind_length,evolution_n,pop_size,v_id):
    global best_code_file, fitness_best_file, fitness_test_file, fitness_pop_file, fitness_evolution_file  # 文件
    global best_code_write, fitness_best_write, fitness_test_write, fitness_pop_write, fitness_evolution_write  # 写入

    global best_ind,best_accuracy,best_coverage,best_fitness,best_c1,best_c2,best_e1,best_e2 #最优个体编码、准确率、覆盖率、正确1、正确2、错误1、错误2
    global view_id

    #全局变量初始化
    view_id=v_id #视角编号

    #记录文件相关变量
    file_list, write_list = csvRecord.create_file(view_id, "ga") #创建文件
    best_code_file, best_code_write = file_list[0], write_list[0]
    fitness_best_file, fitness_best_write = file_list[1], write_list[1]
    fitness_test_file, fitness_test_write = file_list[2], write_list[2]
    fitness_pop_file, fitness_pop_write = file_list[3], write_list[3]
    fitness_evolution_file, fitness_evolution_write = file_list[4], write_list[4]

    #最优个体相关变量
    best_ind = []  # 最优个体编码
    best_fitness, best_accuracy, best_coverage,best_fitness = 0, 0, 0,0  # 最优个体适应度、准确率、覆盖率
    best_c1,best_c2,best_e1,best_e2=0,0,0,0 #最优个体正误情况

    #遗传算法部分参数设置
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))  # 优化目标：单变量，求最大值
    creator.create('Individual', list, fitness=creator.FitnessMax)  # 创建Individual类，继承list

    toolbox = base.Toolbox()  # 实例化一个Toolbox
    toolbox.register('Binary', bernoulli.rvs, 0.5)  # 注册一个Binary的alias，指向scipy.stats中的bernoulli.rvs，概率为0.5
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.Binary,
                     n=ind_length)  # 用tools.initRepeat生成长度为ind_length的individual
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)  # 种群

    toolbox.register('evaluate', evaluate)  # 评估函数
    toolbox.register('mate', tools.cxUniform, indpb=0.5)  # 均匀交叉
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.4)  # 突变
    toolbox.register('roulSel', tools.selRoulette)  # 轮盘赌选择
    CXPB = 0.8  # 交叉概率 有0.8的概率会杂交
    MUTPB = 0.5  # 突变概率 有0.5的概率会发生突变

    #遗传算法初代种群获得
    print("evolution start...")

    csvRecord.set_index(0) #记录文件索引为0

    pop = toolbox.population(pop_size)  # 创建初始种群
    fitnesses = toolbox.evaluate(pop, 0)  # 获取初始种群的适应度 是一个list

    for ind, fit in zip(pop, fitnesses):  # 将对象中对应的元素打包成一个个元组，返回由这些元组组成的列表
        ind.fitness.values = (fit,)

    #进行若干代进化
    for g in range(evolution_n):
        print("         ---Generation %i---" % g)

        fits = [ind.fitness.values[0] for ind in pop]  # 适应度列表
        population_size = len(pop)  # 种群规模
        mean = sum(fits) / population_size  # 种群适应度平均值
        sum2 = sum(x * x for x in fits)  # 种群中个体适应度平方和
        std = abs(sum2 / population_size - mean ** 2) ** 0.5  # 种群适应度标准差

        print(" Min %s" % min(fits))
        print(" Max %s" % max(fits))
        print(" Avg %s" % mean)
        print(" std %s" % std)
        csvRecord.record_line(fitness_evolution_write, [min(fits), max(fits), mean, std])

        #计算本次最优个体在测试集上的情况
        acc, cover,c1,c2,e1,e2,eval_l = viewSeparate.calculate(view_id,best_ind, 2)
        print("测试集：    第", g, "代  ", acc, cover, "\n\n")
        csvRecord.record_line(fitness_test_write, [acc * cover, acc, cover,c1,c2,e1,e2])

        csvRecord.file_flush(file_list) #本代计算结束，将结果写入文件

        #下一代相关计算开始

        csvRecord.set_index(g + 1) #更新索引值

        # 育种选择
        offspring = toolbox.roulSel(pop,pop_size )  # 育种选择 采用轮盘赌选择
        offspring_mate = list(map(toolbox.clone, offspring))  # 不知道怎么命名了，总之就是需要对育种个体进行一次复制，否则修改的是初始种群中的个体，难以达到寻优效果

        # 交叉
        for child1, child2 in zip(offspring_mate[::2], offspring_mate[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)  # child1和child2杂交 生成两个全新个体
                del child1.fitness.values
                del child2.fitness.values

        # 变异
        for mutant in offspring_mate:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)  # 突变，成为新个体
                del mutant.fitness.values

        #评估新种群
        invalid_ind = [ind for ind in offspring_mate if not ind.fitness.valid]  # 选出需要评估适应度个体
        fitnesses = toolbox.evaluate(invalid_ind, g + 1)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        best_pop=[best_ind]
        best_fit_l=[best_fitness]
        for ind, fit in zip(best_pop, best_fit_l):
            ind.fitness.values = (fit,)

        #选出新种群
        pop=best_pop+offspring_mate


        # combine_pop = pop + offspring_mate
        # pop = tools.selBest(combine_pop, pop_size, fit_attr='fitness')  # 选择精英，保持种群规模

    #进化结束 计算最后一代在测试集上的acc
    acc, cover, c1, c2, e1, e2,eval_l = viewSeparate.calculate(view_id,best_ind, 2)
    print("测试集：    第", g, "代  ", acc, cover, "\n\n")
    csvRecord.record_line(fitness_test_write, [acc * cover, acc, cover,c1,c2,e1,e2])

    print("---evolution finish---")

    #关闭文件
    csvRecord.file_close(file_list)  # 关闭文件

    #返回最优个体
    return best_ind
