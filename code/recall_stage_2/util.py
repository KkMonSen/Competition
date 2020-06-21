#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/30 上午 10:54
# @Author  : Monsen
# @Site    :
# @Software: PyCharm

import sys
import copy
import random
import numpy as np
from collections import defaultdict
import pandas as pd


def Get_Recall_S1(fname):
    user_recall = defaultdict(list)
    f = open('../user_data/tmp_data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        user_recall[u].append(i)

    return user_recall

def data_partition(fname, p_fname, v_fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    user_pred = defaultdict(list)
    user_valid_ = defaultdict(list)
    # assume user/item index starting from 1
    if fname is not None:
        f = open('../user_data/tmp_data/%s.txt' % fname, 'r')
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

        for user in User:
            nfeedback = len(User[user])
            if nfeedback < 3:
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = User[user][:-2]
                user_valid[user] = []
                user_valid[user].append(User[user][-2])
                user_test[user] = []
                user_test[user].append(User[user][-1])

    if p_fname is not None:
        p_f = open('../user_data/tmp_data/%s.txt' % p_fname, 'r')
        for line in p_f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            user_pred[u].append(i)

    if v_fname is not None:
        v_f = open('../user_data/tmp_data/%s.txt' % v_fname, 'r')
        for line in v_f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            user_valid_[u].append(i)

    return [user_train, user_valid, user_test, user_pred, user_valid_, usernum, itemnum]


def evaluate(model, dataset, args, sess):
    [train, valid, test, user_pred, user_valid, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)  # item_idx 填充全部商品可用来进行全量召回
        predictions = predictions[0]
        rank = predictions.argsort().argsort()[0]  # 查看用户真正点击的物品被排序到的位置

        valid_user += 1

        if rank < 50:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, sess):
    [train, valid, test, user_pred, user_valid, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)  # 随机抽取用户没点击过的200个物品

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 50:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def predict_result(model, dataset, recall_s1, args, sess, type):
    result = pd.DataFrame()
    [train, valid, test, user_pred, user_valid, usernum, itemnum] = copy.deepcopy(dataset)
    if type == 'pred':
        user_proc = copy.deepcopy(user_pred)
    else:
        user_proc = copy.deepcopy(user_valid)

    pred_item_list = copy.deepcopy(recall_s1)

    for index, u in enumerate(user_proc.keys()):
        if len(pred_item_list[u]) != 101:
            continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in user_proc[u]:
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        item_idx = pred_item_list[u]
        print(index, u, len([u]), len([seq]), len(item_idx))
        predictions = model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]
        recall_prob = np.vstack([np.array(item_idx), predictions])
        recall_prob = recall_prob.transpose()
        recall_prob = pd.DataFrame(data=recall_prob, columns=['item_id', 'prob'])
        # recall_prob = recall_prob.iloc[:50, :]
        recall_prob['user_id'] = u
        result = pd.concat([result, recall_prob])

    if type == 'pred':
        result.to_csv("../user_data/tmp_data/"+args.o_filename+".csv", index=False)
    else:
        result.to_csv("../user_data/tmp_data/valid_recall.csv", index=False)

    return
