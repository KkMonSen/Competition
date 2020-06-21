#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/30 上午 11:47
# @Author  : Monsen
# @Site    : 
# @File    : rec_by_item_cf.py
# @Software: PyCharm

from collections import defaultdict
from tqdm import tqdm
import math
import os
import pandas as pd
import numpy as np
import pickle
import gc
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# gensim 使用cudnn 7.0.5 会与tf1.1以上版本冲突, 训练需要单独分开
# from gensim.models import Word2Vec
import sas_rec
import hash_gene
import rank_by_lgb
import warnings

warnings.filterwarnings('ignore')


def get_u_i_info(train_path, test_path):
	user_info = pd.read_csv(train_path + "/underexpose_user_feat.csv",
							names=['user_id', 'age_level', 'gender', 'city_level'])
	# 填充缺失值
	user_info['age_level'].fillna(-1, inplace=True)
	user_info['gender'].fillna('unknown', inplace=True)
	user_info['city_level'].fillna(-1, inplace=True)

	feat_dummy = pd.get_dummies(user_info['gender'])
	user_info = pd.concat([user_info, feat_dummy], axis=1)
	user_info.drop(['gender'], axis=1, inplace=True)
	user_info.columns = list(user_info.columns[:-3]) + ['gender_F', 'gender_M', 'gender_unknown']

	train_item_df = pd.read_csv(train_path + '/underexpose_item_feat.csv')
	train_item_df.columns = ['item_id'] + ['txt_vec' + str(i) for i in range(128)] + ['img_vec' + str(i) for i in
																					  range(128)]
	train_item_df['txt_vec0'] = train_item_df['txt_vec0'].apply(lambda x: float(x[1:]))
	train_item_df['txt_vec127'] = train_item_df['txt_vec127'].apply(lambda x: float(x[:-1]))
	train_item_df['img_vec0'] = train_item_df['img_vec0'].apply(lambda x: float(x[1:]))
	train_item_df['img_vec127'] = train_item_df['img_vec127'].apply(lambda x: float(x[:-1]))

	# item_txtemb = train_item_df.loc[:, ['item_id'] + ['txt_vec' + str(i) for i in range(128)]]
	# item_imgemb = train_item_df.loc[:, ['item_id'] + ['img_vec' + str(i) for i in range(128)]]
	return user_info


def get_sim_item(whole_df, prepro_df, user_col, item_col, txt_emb, img_emb, use_iif=False):
	# q_time 之前的记录
	df = prepro_df.copy()
	user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()
	pre_user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))

	# 全量数据进行相似度计算
	df = whole_df.copy()
	user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()
	user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))

	user_time_ = df.groupby(user_col)['time'].agg(list).reset_index()  # 引入时间因素
	user_time_dict = dict(zip(user_time_[user_col], user_time_['time']))

	item_user_ = df.groupby(item_col)[user_col].agg(set).reset_index()
	item_user_dict = dict(zip(item_user_[item_col], item_user_[user_col]))

	item_skew = df.groupby(item_col)['time'].agg('skew').reset_index()
	item_skew_dict = dict(zip(item_skew[item_col], item_skew['time']))

	user_skew = df.groupby(user_col)['time'].agg('skew').reset_index()
	user_skew.fillna(6, inplace=True)
	user_skew_dict = dict(zip(user_skew[user_col], user_skew['time']))

	#     # 转为字符串型才能进行训练
	#     doc = df.groupby(['user_id'])['item_id'].agg({list}).reset_index()['list'].values.tolist()

	#     for i in range(len(doc)):
	#         doc[i] = [str(x) for x in doc[i]]

	#     model = Word2Vec(doc, size=128, window=5, min_count=1, sg=0, hs=1, seed=2020)

	#     # 训练结果提取
	#     values = set(df['item_id'].values)
	#     w2v=[]

	#     for v in values:
	#         try:
	#             a = [int(v)]
	#             a.extend(model[str(v)])
	#             w2v.append(a)
	#         except:
	#             pass

	#     out_df = pd.DataFrame(w2v)
	#     out_df.columns = ['item_id'] + ['item_vec'+str(i) for i in range(128)]
	#     i2v_dict = dict(zip(out_df.values[:, 0], list(out_df.values[:, 1:])))

	if use_iif:
		emb_alpha = 1
	else:
		emb_alpha = 1.02

	sim_item = {}
	item_cnt = defaultdict(int)  # 商品被点击次数

	# print('计算二分相似度')
	for item, users in tqdm(item_user_dict.items(), ncols=70, leave=False, unit='b'):
		sim_item.setdefault(item, {})
		for u in users:
			tmp_len = len(user_item_dict[u])
			for relate_item in user_item_dict[u]:
				sim_item[item].setdefault(relate_item, 0)
				sim_item[item][relate_item] += 1 / (
						math.log(len(item_user_dict[relate_item]) + 1) * math.log(len(users) + 1)
						* math.log(tmp_len + 1))

	# print('计算带权重相似度')
	for user, items in tqdm(user_item_dict.items(), ncols=70, leave=False, unit='b'):
		for loc1, item in enumerate(items):
			item_cnt[item] += 1
			sim_item.setdefault(item, {})
			for loc2, relate_item in enumerate(items):
				if item == relate_item:
					continue
				# item_emb = i2v_dict[item]
				# ritem_emb = i2v_dict[relate_item]
				# i2v_sim = 1.0 - (np.dot(item_emb, ritem_emb)/(np.linalg.norm(item_emb)*(np.linalg.norm(ritem_emb))))
				t1 = user_time_dict[user][loc1]  # 点击时间提取
				t2 = user_time_dict[user][loc2]
				t1_skew = item_skew_dict[item]
				t2_skew = item_skew_dict[relate_item]
				sim_item[item].setdefault(relate_item, 0)
				if not use_iif:
					sim_txt, sim_img = 0, 0
				else:
					sim_txt = np.dot(txt_emb[item], txt_emb[relate_item]) / \
							  (np.linalg.norm(txt_emb[item]) * (np.linalg.norm(txt_emb[relate_item])))
					sim_img = np.dot(img_emb[item], img_emb[relate_item]) / \
							  (np.linalg.norm(img_emb[item]) * (np.linalg.norm(img_emb[relate_item])))

				if loc1 - loc2 > 0:
					sim_item[item][relate_item] += (1 * 0.6 * (0.8 ** (loc1 - loc2 - 1)) * (
							1.0 ** abs(t1_skew - t2_skew)) * (1 - (t1 - t2) * 1000) * (
															emb_alpha ** (sim_txt + sim_img)) / (
														math.log(1 + len(items))))  # 逆向 * (0.95 ** i2v_sim)
				else:
					sim_item[item][relate_item] += (1 * 1.0 * (0.8 ** (loc2 - loc1 - 1)) * (
							1.0 ** abs(t2_skew - t1_skew)) * (1 - (t2 - t1) * 1000) * (
															emb_alpha ** (sim_txt + sim_img)) / (
														math.log(1 + len(items))))  # 正向 * (0.95 ** i2v_sim)

	sim_item_corr = sim_item.copy()  # 引入AB的各种被点击次数
	# print('计算最终相似度')
	for i, related_items in tqdm(sim_item.items(), ncols=70, leave=False, unit='b'):
		for j, cij in related_items.items():
			sim_item_corr[i][j] = cij / ((item_cnt[i] * item_cnt[j]) ** 0.2)  # math.log(item_cnt[i] * item_cnt[j] + 1)

	gc.collect()

	return sim_item_corr, pre_user_item_dict, user_skew_dict


def recommend(sim_item_corr, user_item_dict, user_skew_dict, user_id, top_k, item_num):
	# 根据用户的购物偏度进行位置信息衰减
	user_skew = user_skew_dict[user_id]
	skew_alpha = 0.8 + (1 / 190) * (user_skew ** 2)
	rank = {}
	interacted_items = user_item_dict[user_id]
	interacted_items = interacted_items[::-1]
	for loc, i in enumerate(interacted_items):
		for j, wij in sorted(sim_item_corr[i].items(), key=lambda d: d[1], reverse=True)[0:top_k]:
			if j not in interacted_items:
				rank.setdefault(j, 0)
				rank[j] += wij * (skew_alpha ** loc)

	gc.collect()

	return sim_item_corr, sorted(rank.items(), key=lambda d: d[1], reverse=True)[:item_num]


def get_predict(df, pred_col, top_fill, is_train=False):
	# 只保留高于相似度中位数的物品
	# sim_median = df['sim'].median()
	# df = df[df['sim'] >= sim_median]
	top_fill = [int(t) for t in top_fill.split(',')]
	scores = [-1 * i for i in range(1, len(top_fill) + 1)]
	ids = list(df['user_id'].unique())
	fill_df = pd.DataFrame(ids * len(top_fill), columns=['user_id'])
	fill_df.sort_values('user_id', inplace=True)
	fill_df['item_id'] = top_fill * len(ids)
	fill_df[pred_col] = scores * len(ids)
	df = df.append(fill_df)
	df.sort_values(pred_col, ascending=False, inplace=True)
	df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
	df['rank'] = df.groupby('user_id')[pred_col].rank(method='first', ascending=False)
	if 1:
		df = df[df['rank'] <= 101]
	#     df = df.groupby('user_id')['item_id'].apply(lambda x: ','.join([str(i) for i in x])).str.split(',',
	#                                                                                                    expand=True).reset_index()

	gc.collect()

	return df


def get_rank_data(df, pre_df, txt_emb, img_emb, rank_times):
	tmp_df = df.copy()
	# 删除只有单条购物记录的用户
	tmp_filter = tmp_df.groupby(by='user_id')['item_id'].agg('count').reset_index()
	user_filter = tmp_filter[tmp_filter['item_id'] > 1]['user_id'].values
	tmp_df = tmp_df[tmp_df['user_id'].isin(user_filter)]

	# 预测用户删除q_time之后的数据
	tmp_pre_df = pre_df.copy()

	w_recall_data = []
	pre_data = []
	rank_data = []
	# rank_times ： 要构建的rank数据批次
	for i in tqdm(range(rank_times), ncols=70, leave=False, unit='b'):
		tmp_df['rank'] = tmp_df.groupby(by='user_id', as_index=False)['time'].rank(method='first',
																				   ascending=False).values
		rank_data.append(tmp_df[tmp_df['rank'] == 1])
		tmp_df = tmp_df[tmp_df['rank'] != 1][['user_id', 'item_id', 'time']]
		tmp_pre_df['rank'] = tmp_pre_df.groupby('user_id')['time'].rank(method='first', ascending=False)
		tmp_pre_df = tmp_pre_df[tmp_pre_df['rank'] != 1][['user_id', 'item_id', 'time']]
		w_recall_data.append(tmp_df)
		pre_data.append(tmp_pre_df)

		rank_data[i].rename(columns={'rank': 'label'}, inplace=True)
		rank_data[i].fillna(0, inplace=True)
		rank_data[i].columns = ['user_id', 'item_id', 'q_time', 'label']

	#     rank_data = pd.concat(rank_data, sort=False)
	#     rank_data.rename(columns={'rank':'label'}, inplace=True)
	#     rank_data.fillna(0, inplace=True)

	gc.collect()

	return rank_data, w_recall_data, pre_data


def eval_precision(true_data, pre_data, eval_type='recall'):
	click_sum = 0
	user_list = pre_data['user_id'].unique()
	for index, user_id in enumerate(user_list):
		tmp_df = pre_data[pre_data['user_id'] == user_id]
		click_item = true_data[true_data['user_id'] == user_id]['item_id'].values[0]
		try:
			pred_index = np.where(tmp_df['item_id'].values == click_item)[0][0] + 1
			if eval_type != 'recall':
				pred_index = 1
		except:
			pred_index = 0

		click_sum += pred_index

	if eval_type == 'recall':
		score = float((click_sum / 50) / len(user_list))
	else:
		score = float(click_sum / len(user_list))

	return score


def recall(true_df, dict2, whole_df):
	result = 0
	count = 0
	for i in set(true_df['user_id'].unique()):
		if i in dict2 and i in set(whole_df['user_id'].unique()):
			new_item = set()
			tmp_df = true_df[true_df['user_id'] == i]
			tmp_whole_df = whole_df[whole_df['user_id'] == i]
			for k in tmp_df['item_id'].values:
				if k not in tmp_whole_df['item_id'].values:
					new_item.add(k)
			if new_item:
				result += len(new_item & set(dict2[i])) / len(new_item)
				count += 1

	if count == 0:
		return 0
	else:
		return result / count


def GeneRankData(whole_df, q_time_df, recall_data, user_info, sim_corr_dict, is_train=False):
	import time
	whole_df = whole_df.copy()
	q_time = q_time_df.copy()
	recall_data = recall_data.copy()
	# 物品维度：城市、时间交叉
	print('生成物品、用户交互信息')
	hist_item_stat = pd.merge(left=whole_df, right=user_info, on='user_id', how='left')
	hist_item_stat = hist_item_stat.groupby(by='item_id', as_index=False).agg(
		{'city_level': ['mean'], 'age_level': ['mean'],
		 'time': ['count', 'mean', 'median', 'std', 'skew']}).reset_index(
		drop=True)
	hist_item_stat.columns = ['item_id', 'city_mean', 'age_mean', 'time_count', 'time_mean', 'time_median', 'time_std',
							  'time_skew']
	recall_data = pd.merge(left=recall_data, right=hist_item_stat, how='left', on='item_id')
	# 用户维度：物品、时间交叉
	print('生成用户、物品交互信息')
	hist_user_stat = whole_df.groupby(by='user_id', as_index=False).agg(
		{'item_id': 'count', 'time': ['mean', 'median', 'std', 'skew']}).reset_index(drop=True)
	hist_user_stat.columns = ['user_id', 'item_count', 'time_mean', 'time_median', 'time_std', 'time_skew']
	recall_data = pd.merge(left=recall_data, right=hist_user_stat, how='left', on='user_id')
	user_f_click_time = whole_df.groupby('user_id', as_index=False)['time'].agg('max').reset_index(drop=True)
	user_f_click_time = pd.merge(left=user_f_click_time, right=q_time[['user_id', 'q_time']], on='user_id', how='left')
	user_f_click_time['q_time_diff'] = user_f_click_time['q_time'] - user_f_click_time['time']
	#     recall_data = pd.merge(left=recall_data, right=user_info, how='left', on='user_id')
	#     recall_data['gender_unknown'].fillna(1, inplace=True)
	#     recall_data[['age_level', 'city_level']].fillna(-1, inplace=True)
	#     recall_data = pd.merge(left=recall_data, right=item_txtemb, how='left', on='item_id')
	#     recall_data = pd.merge(left=recall_data, right=item_imgemb, how='left', on='item_id')

	if is_train:
		print('recall score:', eval_precision(q_time, recall_data, eval_type='hit_rate'))
		q_time = pd.merge(left=q_time, right=recall_data[['user_id', 'item_id', 'sim', 'rank']],
						  on=['user_id', 'item_id'], how='left')
		pos_user_df = whole_df[whole_df['user_id'].isin(q_time['user_id'].values)]
		pos_user_df = pd.merge(left=pos_user_df, right=user_info, on='user_id', how='left')
		pos_user_df = pos_user_df.groupby(by='item_id', as_index=False).agg(
			{'city_level': ['mean'], 'age_level': ['mean'],
			 'time': ['count', 'mean', 'median', 'std', 'skew']}).reset_index(drop=True)
		pos_user_df.columns = ['item_id', 'city_mean', 'age_mean', 'time_count', 'time_mean', 'time_median', 'time_std',
							   'time_skew']
		recall_rank = pd.merge(left=q_time, right=pos_user_df, how='left', on='item_id')

		pos_item_df = whole_df[whole_df['item_id'].isin(q_time['item_id'].values)]
		pos_item_df = pos_item_df.groupby(by='user_id', as_index=False).agg(
			{'item_id': 'count', 'time': ['mean', 'median', 'std', 'skew']}).reset_index(drop=True)
		pos_item_df.columns = ['user_id', 'item_count', 'time_mean', 'time_median', 'time_std', 'time_skew']
		recall_rank = pd.merge(left=recall_rank, right=pos_item_df, how='left', on='user_id')

		recall_data['label'] = 0

		recall_rank = recall_rank[recall_data.columns]
		recall_data = pd.concat([recall_data, recall_rank])
		recall_data = recall_data.drop_duplicates(subset=['user_id', 'item_id'], keep='last')

	recall_data = pd.merge(left=recall_data, right=user_f_click_time[['user_id', 'q_time_diff']], on='user_id',
						   how='left')
	recall_data.fillna(0, inplace=True)

	gc.collect()

	return recall_data


def add_emb_sim(rec_df, user_item, txt_emb, img_emb):
	user_per_items = {}
	emb_sim_list = []
	for user in user_item.keys():
		tmp_list = user_item[user][-2:]
		if type(tmp_list) == type([]):
			if len(tmp_list) == 2:
				user_per_items[user] = tmp_list
			else:
				for i in range(2 - len(tmp_list)):
					tmp_list.insert(0, 'None')
				user_per_items[user] = tmp_list
		else:
			user_per_items[user] = ['None', tmp_list]

	for user, item in tqdm(rec_df[['user_id', 'item_id']].values):
		tmp_sim = []
		for hist_item in user_per_items[user]:
			if hist_item == 'None':
				sim_txt = 0
				sim_img = 0
			else:
				sim_txt = np.dot(txt_emb[item], txt_emb[hist_item]) / \
						  (np.linalg.norm(txt_emb[item]) * (np.linalg.norm(txt_emb[hist_item])))
				sim_img = np.dot(img_emb[item], img_emb[hist_item]) / \
						  (np.linalg.norm(img_emb[item]) * (np.linalg.norm(img_emb[hist_item])))
			tmp_sim.extend([sim_txt, sim_img])
		emb_sim_list.append(tmp_sim)
	new_feat = ['last_2_t_sim', 'last_2_i_sim', 'last_1_t_sim', 'last_1_i_sim']
	rec_df[new_feat] = pd.DataFrame(emb_sim_list)
	rec_df[new_feat] = pd.DataFrame(emb_sim_list).values
	rec_df.fillna(0, inplace=True)

	return rec_df


def get_l_n_emb_sim(click_data, user_item_dict, item_txtemb, item_imgemb):
	len_3_emb_data = []
	len_5_emb_data = []
	whole_click = click_data.copy()
	# item_emb 相似度
	for user in tqdm(whole_click['user_id'].unique()):
		if len(user_item_dict[user]) >= 5:
			for index in range(2, len(user_item_dict[user]) - 2):
				sim_4 = [user]
				item_t_emb1 = item_txtemb[user_item_dict[user][index]]
				item_i_emb1 = item_imgemb[user_item_dict[user][index]]
				for i_index in [-2, -1, 1, 2]:
					item_t_emb2 = item_txtemb[user_item_dict[user][index + i_index]]
					item_i_emb2 = item_imgemb[user_item_dict[user][index + i_index]]
					sim_txt = np.dot(item_t_emb1, item_t_emb2) / \
							  (np.linalg.norm(item_t_emb1) * (np.linalg.norm(item_t_emb2)))
					sim_img = np.dot(item_i_emb1, item_i_emb2) / \
							  (np.linalg.norm(item_i_emb1) * (np.linalg.norm(item_i_emb2)))
					sim_4.append(sim_txt)
					sim_4.append(sim_img)

				len_5_emb_data.append(sim_4)
		elif len(user_item_dict[user]) >= 3:
			for index in range(1, len(user_item_dict[user]) - 1):
				sim_2 = [user, 0, 0]
				item_t_emb1 = item_txtemb[user_item_dict[user][index]]
				item_i_emb1 = item_imgemb[user_item_dict[user][index]]
				for i_index in [-1, 1]:
					item_t_emb2 = item_txtemb[user_item_dict[user][index + i_index]]
					item_i_emb2 = item_imgemb[user_item_dict[user][index + i_index]]
					sim_txt = np.dot(item_t_emb1, item_t_emb2) / \
							  (np.linalg.norm(item_t_emb1) * (np.linalg.norm(item_t_emb2)))
					sim_img = np.dot(item_i_emb1, item_i_emb2) / \
							  (np.linalg.norm(item_i_emb1) * (np.linalg.norm(item_i_emb2)))
					sim_2.append(sim_txt)
					sim_2.append(sim_img)
				sim_2.extend([0, 0])
				len_3_emb_data.append(sim_2)

	recent_3_emb_df = pd.DataFrame(data=len_3_emb_data,
								   columns=['user_id', 'last2_temb_sim', 'last2_iemb_sim', 'last1_temb_sim',
											'last1_iemb_sim',
											'next1_temb_sim', 'next1_iemb_sim', 'next2_temb_sim', 'next2_iemb_sim', ])
	recent_5_emb_df = pd.DataFrame(data=len_5_emb_data,
								   columns=['user_id', 'last2_temb_sim', 'last2_iemb_sim', 'last1_temb_sim',
											'last1_iemb_sim',
											'next1_temb_sim', 'next1_iemb_sim', 'next2_temb_sim', 'next2_iemb_sim', ])

	recent_5_emb_df = pd.concat([recent_3_emb_df, recent_5_emb_df])
	recent_5_emb_df = (recent_5_emb_df.groupby('user_id').agg('sum') / recent_5_emb_df.groupby('user_id').agg('count'))
	recent_5_emb_df.reset_index(inplace=True)
	recent_5_emb_df.fillna(0, inplace=True)

	return recent_5_emb_df


def GeneRecList(whole_df, prepro_df, user_list, txt_emb, img_emb, topK, sim_num, top50_item, is_train=False):
	whole_df = whole_df.copy()
	recom_item = []
	item_sim_list, user_item, user_skew_dict = get_sim_item(whole_df, prepro_df, 'user_id', 'item_id', txt_emb, img_emb,
															use_iif=False)
	# print('生成推荐列表')
	for i in tqdm(user_list, ncols=70, leave=False, unit='b'):
		sim_item_corr, rank_item = recommend(item_sim_list, user_item, user_skew_dict, i, topK, sim_num)
		for j in rank_item:
			recom_item.append([i, j[0], j[1]])

	recom_df = pd.DataFrame(recom_item, columns=['user_id', 'item_id', 'sim'])
	recall_log = get_predict(recom_df, 'sim', top50_item, is_train)

	gc.collect()

	return sim_item_corr, recom_df, recall_log


def data_preprocessing(train_path, test_path, now_phase):
	pred_user_list = []
	whole_click = pd.DataFrame()
	q_time_df = pd.DataFrame()

	if os.path.exists("../user_data/tmp_data/phase_{}_prepro_data.csv".format(now_phase)):
		whole_click = pd.read_csv("../user_data/tmp_data/phase_{}_whole_data.csv".format(now_phase))
		pre_click = pd.read_csv("../user_data/tmp_data/phase_{}_prepro_data.csv".format(now_phase))
		q_time_df = pd.read_csv("../user_data/tmp_data/phase_{}_user_qtime.csv".format(now_phase))
		pred_user_list = q_time_df['user_id'].values
		# user_list = whole_click['user_id'].unique()
		# user_item = whole_click.groupby('user_id')['item_id'].agg(list).reset_index()
		# user_item = dict(zip(user_item['user_id'], user_item['item_id']))
	else:
		for c in range(now_phase + 1):
			print('phase:', c)
			click_train = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c),
									  header=None, names=['user_id', 'item_id', 'time'])
			click_test = pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_click-{}.csv'.format(c, c),
									 header=None, names=['user_id', 'item_id', 'time'])
			q_time_test = pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_qtime-{}.csv'.format(c, c),
									  header=None, names=['user_id', 'q_time'])

			pred_user_list.extend(click_test['user_id'].unique())
			all_click = click_train.append(click_test)
			whole_click = whole_click.append(all_click)
			q_time_df = q_time_df.append(q_time_test)

		whole_click = whole_click.drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')
		print('before drop:{}'.format(len(whole_click)))
		need_drep_df = whole_click[whole_click['user_id'].isin(q_time_df['user_id'].values)]
		pre_click = whole_click[~whole_click['user_id'].isin(q_time_df['user_id'].values)]

		user_list = q_time_df['user_id'].values
		for user_id in tqdm(user_list, ncols=70, leave=False, unit='b'):
			user_qtime = q_time_df[q_time_df['user_id'] == user_id]['q_time'].values[0]
			save_df = need_drep_df[(need_drep_df['user_id'] == user_id) &
								   (need_drep_df['time'] < user_qtime)]
			pre_click = pre_click.append(save_df)

		print('after drop:{}'.format(len(pre_click)))
		whole_click = whole_click.sort_values('time')
		pre_click = pre_click.sort_values('time')
		whole_click.to_csv("../user_data/tmp_data/phase_{}_whole_data.csv".format(now_phase), index=False)
		pre_click.to_csv("../user_data/tmp_data/phase_{}_prepro_data.csv".format(now_phase), index=False)
		q_time_df.to_csv("../user_data/tmp_data/phase_{}_user_qtime.csv".format(now_phase), index=False)

	top50_click = whole_click['item_id'].value_counts().index[:200].values
	top50_click = ','.join([str(i) for i in top50_click])

	occured_item = pd.Series(data=whole_click['item_id'].unique(), name='item_id')
	# item txt/img embedding
	train_item_df = pd.read_csv(train_path + '/underexpose_item_feat.csv')
	train_item_df.columns = ['item_id'] + ['txt_vec' + str(i) for i in range(128)] + ['img_vec' + str(i) for i in
																					  range(128)]
	train_item_df['txt_vec0'] = train_item_df['txt_vec0'].apply(lambda x: float(x[1:]))
	train_item_df['txt_vec127'] = train_item_df['txt_vec127'].apply(lambda x: float(x[:-1]))
	train_item_df['img_vec0'] = train_item_df['img_vec0'].apply(lambda x: float(x[1:]))
	train_item_df['img_vec127'] = train_item_df['img_vec127'].apply(lambda x: float(x[:-1]))
	occured_item = pd.merge(occured_item, train_item_df, on='item_id', how='left')
	occured_item.fillna(0, inplace=True)
	item_txtemb = occured_item.loc[:, ['item_id'] + ['txt_vec' + str(i) for i in range(128)]]
	item_imgemb = occured_item.loc[:, ['item_id'] + ['img_vec' + str(i) for i in range(128)]]
	item_txtemb = dict(zip(item_txtemb['item_id'], item_txtemb.loc[:, ['txt_vec' + str(i) for i in range(128)]].values))
	item_imgemb = dict(zip(item_imgemb['item_id'], item_imgemb.loc[:, ['img_vec' + str(i) for i in range(128)]].values))

	gc.collect()

	return whole_click, pre_click, q_time_df, pred_user_list, top50_click, item_txtemb, item_imgemb


def get_test_rec(whole_click, prepro_click, pred_user_list, item_txtemb, item_imgemb, top50_click, now_phase):
	# 获取预测集推荐集
	if os.path.exists("../user_data/tmp_data/phase_{}_recall_stage_1.csv".format(now_phase)):
		recall_log = pd.read_csv("../user_data/tmp_data/phase_{}_recall_stage_1.csv".format(now_phase))
	else:
		sim_corr_dict, _, recall_log = GeneRecList(whole_click, prepro_click, pred_user_list, item_txtemb, item_imgemb,
												   500, 101,
												   top50_click)
		recall_log.to_csv("../user_data/tmp_data/phase_{}_recall_stage_1.csv".format(now_phase), index=False)

	return recall_log


# 对预测集进行重排
def get_stage2_recall(recall_log):
	ori_result = recall_log
	gene_result = pd.read_csv("../user_data/tmp_data/result.csv")

	with open("../user_data/tmp_data/index_2_user_dict.pkl", 'rb') as f:
		index_2_user = pickle.load(f)

	with open("../user_data/tmp_data/index_2_item_dict.pkl", 'rb') as f:
		index_2_item = pickle.load(f)

	# 映射回原来id
	gene_result['user_id'] = gene_result['user_id'].apply(lambda x: index_2_user[x])
	gene_result['item_id'] = gene_result['item_id'].apply(lambda x: index_2_item[x])

	gene_result = pd.merge(left=ori_result, right=gene_result[['user_id', 'item_id', 'prob']],
						   on=['user_id', 'item_id'], how='left')

	gene_result['rank_prob'] = gene_result.groupby('user_id')['prob'].rank(method='first', ascending=False)
	gene_result['rank_sim'] = gene_result.groupby('user_id')['sim'].rank(method='first', ascending=False)

	gene_result['rank_ensemble'] = (0.3 * gene_result['rank_prob']) + (0.7 * gene_result['rank_sim'])

	gene_result['rank'] = gene_result.groupby('user_id')['rank_ensemble'].rank(method='first', ascending=True)

	gene_result = gene_result[gene_result['rank'] <= 80]

	final_recall_log = gene_result[['user_id', 'item_id', 'sim', 'rank']]

	return final_recall_log


def gene_test_data(whole_click, q_time_df, final_recall_log, user_info, item_txtemb, item_imgemb):
	if os.path.exists("../user_data/tmp_data/test_data.csv"):
		final_recall_log = pd.read_csv("../user_data/tmp_data/test_data.csv")
	else:
		# 生成训练集排序数据
		final_recall_log = GeneRankData(whole_click, q_time_df, final_recall_log, user_info, sim_corr_dict=None)
		user_item_ = whole_click.groupby('user_id')['item_id'].agg(list).reset_index()
		user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
		recent_emb_sim = get_l_n_emb_sim(whole_click, user_item_dict, item_txtemb, item_imgemb)
		final_recall_log = pd.merge(final_recall_log, recent_emb_sim, on='user_id', how='left')
		final_recall_log = add_emb_sim(final_recall_log, user_item_dict, item_txtemb, item_imgemb)
		final_recall_log.to_csv("../user_data/tmp_data/test_data.csv", index=False)

	return final_recall_log


def pos_sim_fill(data, sim_item_corr, user_item_dict, user_skew_dict):
	# 根据用户的购物偏度进行位置信息衰减
	fill_list = []
	for user_id, item_id in tqdm(data, ncols=70, leave=False, unit='b'):
		tmp_sim = 0
		user_skew = user_skew_dict[user_id]
		skew_alpha = 0.8 + (1 / 190) * (user_skew ** 2)
		interacted_items = user_item_dict[user_id]
		interacted_items = interacted_items[::-1]
		if item_id not in sim_item_corr:
			fill_list.append(0)
			continue
		i_list = set(interacted_items).intersection(set(sim_item_corr[item_id].keys()))
		for loc, i in enumerate(interacted_items):
			if i in i_list:
				tmp_sim += sim_item_corr[i][item_id] * (skew_alpha ** loc)
		fill_list.append(tmp_sim)

	return fill_list


def gene_train_data(whole_click, pre_click, item_txtemb, item_imgemb, now_phase, top50_click, user_info, get_data_batch=1):
	# # 排序训练集生成
	if os.path.exists("../user_data/tmp_data/phase_{}_train_data.csv".format(now_phase)):
		recall_log = pd.read_csv("../user_data/tmp_data/phase_{}_train_data.csv".format(now_phase))
	else:
		# get_data_batch为要获取多少批训练数据
		uline_rank_data, w_recall_data, w_pre_click = get_rank_data(whole_click, pre_click, item_txtemb, item_imgemb,
																	get_data_batch)
		recall_log = pd.DataFrame()
		for index, ori_data in enumerate(w_recall_data):
			user_list = ori_data['user_id'].unique()
			# item_num = len(ori_data['item_id'].unique())
			# 获取排序训练数据
			ori_data = ori_data.drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')
			ori_data = ori_data.sort_values('time')
			sim_corr_dict, sim_item_df, tmp_recall_log = GeneRecList(ori_data, w_pre_click[index], user_list,
																	 item_txtemb, item_imgemb,
																	 500, 101, top50_click, is_train=True)
			tmp_log = GeneRankData(w_pre_click[index], uline_rank_data[index], tmp_recall_log, user_info,
								   sim_corr_dict=None, is_train=True)
			recall_log = pd.concat([recall_log, tmp_log])
			user_item = ori_data.groupby('user_id')['item_id'].agg(list).reset_index()
			user_item_dict = dict(zip(user_item['user_id'], user_item['item_id']))

			user_skew = ori_data.groupby('user_id')['time'].agg('skew').reset_index()
			user_skew_dict = dict(zip(user_skew['user_id'], user_skew['time']))

			f_data = recall_log[recall_log['sim'] == 0][['user_id', 'item_id']].values
			fill_list = pos_sim_fill(f_data, sim_corr_dict, user_item_dict, user_skew_dict)
			recall_log.loc[recall_log['sim'] == 0, 'sim'] = fill_list

			user_item_ = ori_data.groupby('user_id')['item_id'].agg(list).reset_index()
			user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
			recent_emb_sim = get_l_n_emb_sim(ori_data, user_item_dict, item_txtemb, item_imgemb)
			recall_log = pd.merge(recall_log, recent_emb_sim, on='user_id', how='left')
			recall_log = add_emb_sim(recall_log, user_item_dict, item_txtemb, item_imgemb)

		# 设置正负样本比例1：6
		pos_data = recall_log[recall_log['label'] == 1]
		nag_data = recall_log[recall_log['label'] == 0].sample(n=len(pos_data) * 6)
		recall_log = pd.concat([pos_data, nag_data])
		recall_log.to_csv("../user_data/tmp_data/phase_{}_train_data.csv".format(now_phase), index=False)

	return recall_log


def item_cf_rec_and_rank(train_path, test_path, now_phase, args):
	def show_log(x): print("-" * 50, x, "-" * 50)

	# 1、用户信息读取及数据预处理
	show_log("Preprocessing")
	user_info = get_u_i_info(train_path, test_path)
	whole_click, pre_cilck, q_time_df, pred_user_list, top50_click, item_txtemb, item_imgemb = \
		data_preprocessing(train_path, test_path, now_phase)

	# 2、使用item_cf进行初召回
	show_log("Recall Stage_1")
	test_recall_log = get_test_rec(pre_cilck, pre_cilck, pred_user_list, item_txtemb, item_imgemb, top50_click,
								   now_phase)

	# 3、id哈希映射
	show_log("Hash Reflecting")
	hash_gene.get_hash_ref(train_path, test_path, now_phase)

	# 4、使用sas rec 进行二次召回筛选
	show_log("Recall Stage_2")
	sas_rec.train_sasrec(args)
	final_recall_log = get_stage2_recall(test_recall_log)

	# 5、生成排序数据
	show_log("Generate Rank Data")
	train_dataset = gene_train_data(pre_cilck, pre_cilck, item_txtemb, item_imgemb, now_phase, top50_click,
									user_info, get_data_batch=1)
	test_data = gene_test_data(pre_cilck, q_time_df, final_recall_log, user_info, item_txtemb, item_imgemb)

	# 6、排序及生成最终结果
	show_log("Generate final result")
	model, rank_data = rank_by_lgb.train_rank_model(train_dataset, test_data)
	rank_by_lgb.get_final_result(rank_data, pred_user_list, top50_click)
