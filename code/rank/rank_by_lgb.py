#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/30 上午 11:47
# @Author  : Monsen
# @Site    : 
# @File    : rank_by_lgb.py
# @Software: PyCharm

import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
import numpy as np


# 训练模型
def train_rank_model(train_dataset, test_data):
	# result_df = pd.DataFrame()
	pos_data = train_dataset[train_dataset['label'] == 1]
	nag_data = train_dataset[train_dataset['label'] == 0].sample(n=len(pos_data) * 3)
	train_dataset = pd.concat([pos_data, nag_data])

	features = [x for x in train_dataset.columns if
				x not in ['user_id', 'item_id', 'prob', 'rank', 'label']]
	# 归一化
	# scaler = MinMaxScaler()
	# test_data[features] = scaler.fit_transform(test_data[features])

	# train_dataset[features] = scaler.transform(train_dataset[features])

	train_dataset = train_dataset.sample(frac=1.0)

	# # 对user进行分桶
	all_user = train_dataset['user_id'].unique()
	train_data = train_dataset[train_dataset['user_id'].isin(all_user[:int(len(all_user) * 0.8)])]
	valid_data = train_dataset[~train_dataset['user_id'].isin(all_user[:int(len(all_user) * 0.8)])]
	llf = lgb.LGBMClassifier(num_leaves=8
							 , max_depth=3
							 , learning_rate=0.01
							 , n_estimators=3000
							 , class_weight={0: 1, 1: 3}
							 , objective='binary'
							 , n_jobs=-1
							 , reg_alpha=0.1
							 , reg_lambda=0.1
							 , silent=False
							 , verbose=0
							 , metric='auc')
	llf.fit(train_data[features], train_data['label'], eval_set=[(valid_data[features], valid_data['label'])],
			eval_metric='auc')

	test_data['label'] = llf.predict_proba(test_data[features])[:, 1]
	test_data.to_csv("../user_data/tmp_data/lgb_result.csv", index=False)
	return llf, test_data


def get_final_result(rank_data, user_list, top50_click):
	# 生成最终排序结果
	result_logs = dict()

	rank_data.sort_values('label', ascending=False, inplace=True)
	for row in tqdm(rank_data[['user_id', 'item_id']].values, ncols=70, leave=False, unit='b'):
		result_logs.setdefault(row[0], [])
		if len(result_logs[row[0]]) < 50:
			if row[1] not in result_logs[row[0]]:
				result_logs[row[0]].append(row[1])

	rec_dict = dict()
	for u in tqdm(set(user_list), ncols=70, leave=False, unit='b'):
		if u in result_logs:
			lenth = len(np.unique(result_logs[u]))
			if lenth < 50:
				rec_dict[u] = result_logs[u] + [int(x) for x in top50_click.split(',') if x not in result_logs[u] and
												x not in result_logs[u]][:50 - lenth]
			else:
				rec_dict[u] = result_logs[u]
		else:
			rec_dict[u] = [x for x in top50_click.split(',')][:50]

	result = pd.DataFrame(data=rec_dict).T
	result.sort_index(inplace=True)
	result.to_csv("../prediction_result/result_rank_f.csv", header=False)
