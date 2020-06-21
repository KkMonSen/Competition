#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/30 上午 10:52
# @Author  : Monsen
# @Site    : 
# @File    : sas_rec.py
# @Software: PyCharm

import os
import time
import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
import math
from util import *


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def train_sasrec(n_args):
    if not os.path.exists("../../prediction_result/"+n_args.o_filename+".csv"):
        if not os.path.isdir(n_args.dataset + '_' + n_args.train_dir):
            os.makedirs(n_args.dataset + '_' + n_args.train_dir)
        with open(os.path.join(n_args.dataset + '_' + n_args.train_dir, 'args.txt'), 'w') as f:
            f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(n_args).items(), key=lambda x: x[0])]))
        f.close()

        dataset = data_partition(n_args.dataset, n_args.p_dataset, None)
        recall_s1 = Get_Recall_S1(n_args.recall_ds)
        # recall_v = Get_Recall_S1(n_args.recall_v)
        [user_train, user_valid, user_test, user_pred, user_valid_, usernum, itemnum] = dataset
        num_batch = math.ceil(len(user_train) / n_args.batch_size)
        cc = 0.0
        for u in user_train:
            cc += len(user_train[u])
        print('average sequence length: %.2f' % (cc / len(user_train)))

        f = open(os.path.join(n_args.dataset + '_' + n_args.train_dir, 'log.txt'), 'w')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        sampler = WarpSampler(user_train, usernum, itemnum, batch_size=n_args.batch_size, maxlen=n_args.maxlen, n_workers=4)
        model = Model(usernum, itemnum, n_args)

        if not os.listdir("../user_data/model_data/"):
            sess.run(tf.global_variables_initializer())
            T = 0.0
            t0 = time.time()
            try:
                for epoch in range(1, n_args.num_epochs + 1):
                    for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                        u, seq, pos, neg = sampler.next_batch()
                        auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                                {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                                 model.is_training: True})

                    if epoch % 20 == 0:
                        t1 = time.time() - t0
                        T += t1
                        print('Evaluating')
                        t_test = evaluate(model, dataset, n_args, sess)
                        t_valid = evaluate_valid(model, dataset, n_args, sess)
                        print('')
                        print('epoch:%d, time: %f(s), valid (NDCG@50: %.4f, HR@10: %.4f), test (NDCG@50: %.4f, HR@10: %.4f)' % (
                        epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
                        f.write(str(t_valid) + ' ' + str(t_test) + '\n')
                        f.flush()
                        t0 = time.time()
                saver = tf.train.Saver()
                saver.save(sess, "../user_data/model_data/sasrec_model.ckpt")
                predict_result(model, dataset, recall_s1, n_args, sess, type='pred')
                # predict_result(model, dataset, recall_v, args, sess, type='valid')

            except:
                sampler.close()
                f.close()
                exit(1)
        else:
            saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess, "../user_data/model_data/sasrec_model.ckpt")
                predict_result(model, dataset, recall_s1, n_args, sess, type='pred')
                # predict_result(model, dataset, recall_v, args, sess, type='valid')

        f.close()
        sampler.close()
        print("Done")
        # exit(0)


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
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=101, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.001, type=float)

    args = parser.parse_args()
    train_sasrec(args)

