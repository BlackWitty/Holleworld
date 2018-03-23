# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:19:36 2018

@author: BlackCristle
"""

import pandas as pd
import numpy as np
import datetime as date
import datetime as dt

#先导入数据

off_train = pd.read_csv("data/ccf_offline_stage1_train.csv",header = 0)
off_train.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']

off_test = pd.read_csv("data/ccf_offline_stage1_test_revised.csv",header = 0)
off_test.columns  = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received']

on_train = pd.read_csv("data/ccf_online_stage1_train.csv",header=0)
on_train.columns = ['user_id','merchant_id','action','coupon_id','discount_rate','date_received','date']

# 按照时间划分训练集和测试集
# 滑窗法划分
"""
将2016年1月1日到4月13日的数据提取特征，利用4月14日的到5月14日的作为测试集
将2月1日到5月14日的作为数据集提取特征，利用5月15日6月15日的作为测试集
将3月15日到6月30日作为数据集提取特征，再测试7月1日到7月31日的数据
dataset用来做测试集，feature用来做训练集
"""
#数据集3的特征为 取 线上数据中领券和用券日期大于3月15日和小于6月30日的
#将3月15日到6月30日作为数据集提取特征，再测试7月1日到7月31日的数据
#使数据集3等于test集
""""
dataset里面只有接收优惠券的记录的，无消费记录，可用于预测
feature里面存的是优惠券使用日期或接收优惠券的时间介于3月15日到6月30之间的记录
"""
dataset3 = off_test
feature3 = off_train[((off_train.date>='20160315')&(off_train.date<='20160630')|((off_train.date=='null')&(off_train.date_received>='20160315')&(off_train.date_received<='20160630')))]

#提取数据集2的测试集
#将2月1日到5月14日的作为数据集提取特征，利用5月15日6月15日的作为测试集
""""
dataset里面只存放优惠券接收日期介于5月15日到6月15之间的记录
feature里面存的是优惠券使用日期或接收优惠券的时间介于2月1日到5月14之间的记录
"""
dataset2 = off_train[((off_train.date_received>='20160515')&(off_train.date_received<='20160615'))]

feature2 = off_train[(off_train.date>='20160201')&(off_train.date<='20160514')|((off_train.date=='null')&(off_train.date_received>='20160201')&(off_train.date_received<='20160514'))]
""""
dataset里面只存放优惠券接收日期介于4月14日到5月14之间的记录
feature里面存的是优惠券使用日期或接收优惠券的时间介于1月1日到4月13之间的记录
"""
dataset1 = off_train[(off_train.date_received>='201604014')&(off_train.date_received<='20160514')]
feature1 = off_train[(off_train.date>='20160101')&(off_train.date<='20160413')|((off_train.date=='null')&(off_train.date_received>='20160101')&(off_train.date_received<='20160413'))]
"""
# 提取特征：
    用户领取的所有优惠券数目
    ◦用户领取的特定优惠券数目
    ◦用户此次之后/前领取的所有优惠券数目
    ◦用户此次之后/前领取的特定优惠券数目
    ◦用户上/下一次领取的时间间隔
    ◦用户领取特定商家的优惠券数目
    ◦用户领取的不同商家数目
    ◦用户当天领取的优惠券数目
    ◦用户当天领取的特定优惠券数目
    ◦用户领取的所有优惠券种类数目
    ◦商家被领取的优惠券数目
    ◦商家被领取的特定优惠券数目
    ◦商家被多少不同用户领取的数目
    ◦商家发行的所有优惠券种类数目
"""
# 用户收到的优惠券总和
t = dataset3[['user_id']]
t['this_month_user_received_all_coupon_count'] = 1
 #将t按照用户id进行分组，然后统计所有用户收取的优惠券数目,并初始化一个索引值
t = t.groupby('user_id').agg('sum').reset_index()
# 用户收到特定优惠券的总和
t1 = dataset3[['user_id','coupon_id']]
t1['this_month_user_coupon_received_count'] = 1
t1 = t1.groupby(['user_id','coupon_id']).agg('sum').reset_index()

# 用户此次之前或之后领使用优惠券的时间
# lambda x:':'.join(x)  是添加冒号并在后面去加字符
# 将接收时间的一组按着':'分开，这样就可以计算接受了优惠券的数量,apply是合并
# 最大接受的日期max_date_received/min_date_received
t2 = dataset3[['user_id','coupon_id','date_received']]
t2.date_received = t2.date_received.astype('str')
t2 = t2.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
t2['receive_number'] = t2.date_received.apply(lambda s:len(s.split(':')))
t2 = t2[t2.receive_number>1]
t2['max_date_received'] = t2.date_received.apply(lambda s:max([int (d) for d in s.split(':')]))
t2['min_date_received'] = t2.date_received.apply(lambda s:min([int (d) for d in s.split(':')]))
t2 = t2[['user_id','coupon_id','max_date_received','min_date_received']]


# 将两表融合只保留左表数据,这样得到的表，相当于保留了最近接收时间和最远接受时间
t3 = dataset3[['user_id','coupon_id','date_received']]
t3 = pd.merge(t2,t3,on=['user_id','coupon_id'],how='left')
t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received-t3.date_received.astype(int)