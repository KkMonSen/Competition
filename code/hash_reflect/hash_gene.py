#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/30 上午 10:59
# @Author  : Monsen
# @Site    : 
# @File    : hash_gene.py
# @Software: PyCharm

import pandas as pd
from tqdm import tqdm
import os
import pickle as pkl


def get_hash_ref(train_path, test_path, now_phase):
    pred_user_list = []
    whole_click = pd.DataFrame()
    q_time_df = pd.DataFrame()

    if os.path.exists("../user_data/tmp_data/phase_{}_prepro_data.csv".format(now_phase)):
        whole_click = pd.read_csv("../user_data/tmp_data/phase_{}_whole_data.csv".format(now_phase))
        q_time_df = pd.read_csv("../user_data/tmp_data/phase_{}_user_qtime.csv".format(now_phase))
        # user_list = q_time_df['user_id'].values
        # user_item = whole_click.groupby('user_id')['item_id'].agg(list).reset_index()
    # user_item = dict(zip(user_item['user_id'], user_item['item_id']))
    else:
        for c in range(now_phase + 1):
            print('phase:', c)
            click_train = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c), header=None,
                                      names=['user_id', 'item_id', 'time'])
            click_test = pd.read_csv(test_path + '/underexpose_test_click-{}.csv'.format(c, c), header=None,
                                     names=['user_id', 'item_id', 'time'])
            q_time_test = pd.read_csv(test_path + '/underexpose_test_qtime-{}.csv'.format(c, c), header=None,
                                      names=['user_id', 'q_time'])

            pred_user_list.extend(click_test['user_id'].unique())
            all_click = click_train.append(click_test)
            whole_click = whole_click.append(all_click)
            q_time_df = q_time_df.append(q_time_test)

        whole_click = whole_click.drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')
        print('before drop:{}'.format(len(whole_click)))
        need_drep_df = whole_click[whole_click['user_id'].isin(q_time_df['user_id'].values)]
        whole_click = whole_click[~whole_click['user_id'].isin(q_time_df['user_id'].values)]

        user_list = q_time_df['user_id'].values
        for user_id in tqdm(user_list, ncols=70, leave=False, unit='b'):
            user_qtime = q_time_df[q_time_df['user_id'] == user_id]['q_time'].values[0]
            save_df = need_drep_df[(need_drep_df['user_id'] == user_id) &
                                   (need_drep_df['time'] < user_qtime)]
            whole_click = whole_click.append(save_df)

        print('after drop:{}'.format(len(whole_click)))
        whole_click = whole_click.sort_values('time')
        whole_click.to_csv("../user_data/tmp_data/phase_{}_prepro_data.csv".format(now_phase), index=False)
        q_time_df.to_csv("../user_data/tmp_data/phase_{}_user_qtime.csv".format(now_phase), index=False)

    # # 用户、物品哈希映射
    user_2_index_dict = {}
    index_2_user_dict = {}

    item_2_index_dict = {}
    index_2_item_dict = {}

    for index, user_id in enumerate(set(whole_click['user_id'].values)):
        index_2_user_dict[index + 1] = user_id
        user_2_index_dict[user_id] = index + 1

    for index, item_id in enumerate(set(whole_click['item_id'].values)):
        index_2_item_dict[index + 1] = item_id
        item_2_index_dict[item_id] = index + 1

    with open("../user_data/tmp_data/user_2_index_dict.pkl", 'wb') as u_i_dict:
        pkl.dump(user_2_index_dict, u_i_dict)

    with open("../user_data/tmp_data/index_2_user_dict.pkl", 'wb') as i_u_dict:
        pkl.dump(index_2_user_dict, i_u_dict)

    with open("../user_data/tmp_data/item_2_index_dict.pkl", 'wb') as it_i_dict:
        pkl.dump(item_2_index_dict, it_i_dict)

    with open("../user_data/tmp_data/index_2_item_dict.pkl", 'wb') as i_it_dict:
        pkl.dump(index_2_item_dict, i_it_dict)

    whole_click.sort_values(by=['user_id', 'time'], ascending=True, inplace=True)

    # # 提取出预测集与训练集
    pred_user = q_time_df['user_id'].unique()
    pred_user_df = whole_click[whole_click['user_id'].isin(pred_user)].copy()
    diff_user_df = whole_click[~whole_click['user_id'].isin(pred_user)].copy()

    pred_user_df['rank'] = pred_user_df.groupby('user_id')['time'].rank(method='first', ascending=False).values

    pred_user_df['user_id'] = pred_user_df['user_id'].apply(lambda x: user_2_index_dict[x])
    pred_user_df['item_id'] = pred_user_df['item_id'].apply(lambda x: item_2_index_dict[x])

    diff_user_df['user_id'] = diff_user_df['user_id'].apply(lambda x: user_2_index_dict[x])
    diff_user_df['item_id'] = diff_user_df['item_id'].apply(lambda x: item_2_index_dict[x])

    # # 先保留全量预测集数据
    pred_user_df.sort_values(by=['user_id', 'time'], ascending=[True, False], inplace=True)
    pred_user_df[['user_id', 'item_id']].to_csv("../user_data/tmp_data/pred_serial.txt", sep=' ', index=False, header=False)
    pred_user_df = pred_user_df[pred_user_df['rank'] > 1]
    whole_click = pd.concat([diff_user_df, pred_user_df[['user_id', 'item_id', 'time']]])
    whole_click.sort_values(by=['user_id', 'time'], ascending=True, inplace=True)
    whole_click[['user_id', 'item_id']].to_csv("../user_data/tmp_data/sas_serial.txt", sep=' ', index=False, header=False)

    # stage_1 user/item id 映射
    recall_stage = pd.read_csv("../user_data/tmp_data/phase_{}_recall_stage_1.csv".format(now_phase))

    recall_stage['user_id'] = recall_stage['user_id'].apply(lambda x: user_2_index_dict[x])
    recall_stage['item_id'] = recall_stage['item_id'].apply(lambda x: item_2_index_dict[x])

    recall_stage[['user_id', 'item_id']].to_csv("../user_data/tmp_data/recall_stage1.txt", sep=' ', index=False, header=False)
