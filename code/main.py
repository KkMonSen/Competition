#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/30 下午 05:35
# @Author  : Monsen
# @Site    : 
# @File    : main.py
# @Software: PyCharm

import rec_by_item_cf
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', required=True)
	parser.add_argument('--p_dataset', required=True)
	parser.add_argument('--v_dataset', required=True)
	parser.add_argument('--recall_v', required=True)
	parser.add_argument('--recall_ds', required=True)
	parser.add_argument('--o_filename', required=True)
	parser.add_argument('--train_dir', required=True)
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--lr', default=0.002, type=float)
	parser.add_argument('--maxlen', default=50, type=int)
	parser.add_argument('--hidden_units', default=128, type=int)
	parser.add_argument('--num_blocks', default=2, type=int)
	parser.add_argument('--num_epochs', default=41, type=int)
	parser.add_argument('--num_heads', default=1, type=int)
	parser.add_argument('--dropout_rate', default=0.2, type=float)
	parser.add_argument('--l2_emb', default=0.01, type=float)
	args = parser.parse_args()

	now_phase = 9
	train_path = '../data/underexpose_train'
	test_path = '../data/underexpose_test'

	rec_by_item_cf.item_cf_rec_and_rank(train_path, test_path, now_phase, args)
