"""
Tensorflow related part
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from src import param


class TFParts(object):
    '''
    TensorFlow-related things.
    Keep TensorFlow-related components in a neat shell.
    '''

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg):
        self._num_rels = num_rels
        self._num_cons = num_cons
        self._dim = dim  # dimension of both relation and ontology.
        self._batch_size = batch_size
        self._neg_per_positive = neg_per_positive
        self._epoch_loss = 0
        self._p_neg = 1
        self._p_psl = 0.2
        self._soft_size = 1
        self._prior_psl = 0

    def build_basics(self):
        tf.reset_default_graph()
        with tf.variable_scope("graph", initializer=tf.truncated_normal_initializer(0, 0.3)):
            # Variables (matrix of embeddings/transformations)
            self._ht = ht = tf.get_variable(
                name='ht',  # for t AND h
                shape=[self.num_cons, self.dim],
                dtype=tf.float32)

            self._r = r = tf.get_variable(
                name='r',
                shape=[self.num_rels, self.dim],
                dtype=tf.float32)

            self._A_h_index = A_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='A_h_index')
            self._A_r_index = A_r_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='A_r_index')
            self._A_t_index = A_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='A_t_index')

            # for uncertain graph
            self._A_w = tf.placeholder(
                dtype=tf.float32,
                shape=[self.batch_size],
                name='_A_w')

            self._A_neg_hn_index = A_neg_hn_index = tf.placeholder(
                dtype=tf.int64,
                shape=(self.batch_size, self._neg_per_positive),
                name='A_neg_hn_index')
            self._A_neg_rel_hn_index = A_neg_rel_hn_index = tf.placeholder(
                dtype=tf.int64,
                shape=(self.batch_size, self._neg_per_positive),
                name='A_neg_rel_hn_index')
            self._A_neg_t_index = A_neg_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=(self.batch_size, self._neg_per_positive),
                name='A_neg_t_index')
            self._A_neg_h_index = A_neg_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=(self.batch_size, self._neg_per_positive),
                name='A_neg_h_index')
            self._A_neg_rel_tn_index = A_neg_rel_tn_index = tf.placeholder(
                dtype=tf.int64,
                shape=(self.batch_size, self._neg_per_positive),
                name='A_neg_rel_tn_index')
            self._A_neg_tn_index = A_neg_tn_index = tf.placeholder(
                dtype=tf.int64,
                shape=(self.batch_size, self._neg_per_positive),
                name='A_neg_tn_index')

            # no normalization
            self._h_batch = tf.nn.embedding_lookup(ht, A_h_index)
            self._t_batch = tf.nn.embedding_lookup(ht, A_t_index)
            self._r_batch = tf.nn.embedding_lookup(r, A_r_index)

            self._neg_hn_con_batch = tf.nn.embedding_lookup(ht, A_neg_hn_index)
            self._neg_rel_hn_batch = tf.nn.embedding_lookup(r, A_neg_rel_hn_index)
            self._neg_t_con_batch = tf.nn.embedding_lookup(ht, A_neg_t_index)
            self._neg_h_con_batch = tf.nn.embedding_lookup(ht, A_neg_h_index)
            self._neg_rel_tn_batch = tf.nn.embedding_lookup(r, A_neg_rel_tn_index)
            self._neg_tn_con_batch = tf.nn.embedding_lookup(ht, A_neg_tn_index)

            # psl batches
            self._soft_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._soft_size],
                name='soft_h_index')
            self._soft_r_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._soft_size],
                name='soft_r_index')
            self._soft_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._soft_size],
                name='soft_t_index')

            # for uncertain graph and psl
            self._soft_w = tf.placeholder(
                dtype=tf.float32,
                shape=[self._soft_size],
                name='soft_w_lower_bound')

            self._soft_h_batch = tf.nn.embedding_lookup(ht, self._soft_h_index)
            self._soft_t_batch = tf.nn.embedding_lookup(ht, self._soft_t_index)
            self._soft_r_batch = tf.nn.embedding_lookup(r, self._soft_r_index)


    def build_optimizer(self):
        self._A_loss = tf.add(self.main_loss, self.psl_loss)

        # Optimizer
        self._lr = lr = tf.placeholder(tf.float32)
        self._opt = opt = tf.train.AdamOptimizer(lr)

        # This can be replaced by
        # self._train_op_A = train_op_A = opt.minimize(A_loss)
        self._gradient = gradient = opt.compute_gradients(self._A_loss)  # splitted for debugging

        self._train_op = opt.apply_gradients(gradient)

        # Saver
        self._saver = tf.train.Saver(max_to_keep=2)

    def build(self):
        self.build_basics()
        self.define_main_loss()  # abstract method. get self.main_loss
        self.define_psl_loss()  # abstract method. get self.psl_loss
        self.build_optimizer()

    def compute_psl_loss(self):
        self.prior_psl0 = tf.constant(self._prior_psl, tf.float32)
        self.psl_error_each = tf.square(tf.maximum(self._soft_w + self.prior_psl0 - self.psl_prob, 0))
        self.psl_mse = tf.reduce_mean(self.psl_error_each)
        self.psl_loss = self.psl_mse * self._p_psl

    @property
    def num_cons(self):
        return self._num_cons

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def neg_batch_size(self):
        return self._neg_per_positive * self._batch_size


class UKGE_logi_TF(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg)
        self.build()

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self._htr = htr = tf.reduce_sum(
            tf.multiply(self._r_batch, tf.multiply(self._h_batch, self._t_batch, "element_wise_multiply"),
                        "r_product"),
            1)
        self._f_prob_h = f_prob_h = tf.sigmoid(self.w * htr + self.b)  # logistic regression
        self._f_score_h = f_score_h = tf.square(tf.subtract(f_prob_h, self._A_w))

        self._f_prob_hn = f_prob_hn = tf.sigmoid(self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_hn_batch, tf.multiply(self._neg_hn_con_batch, self._neg_t_con_batch)), 2)
        ) + self.b)
        self._f_score_hn = f_score_hn = tf.reduce_mean(tf.square(f_prob_hn), 1)

        self._f_prob_tn = f_prob_tn = tf.sigmoid(self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_tn_batch, tf.multiply(self._neg_h_con_batch, self._neg_tn_con_batch)), 2)
        ) + self.b)
        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(f_prob_tn), 1)

        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size

    def define_psl_loss(self):
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts


class UKGE_rect_TF(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg)
        self.reg_scale = reg_scale
        self.build()

    # override
    def define_main_loss(self):
        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self._htr = htr = tf.reduce_sum(
            tf.multiply(self._r_batch, tf.multiply(self._h_batch, self._t_batch, "element_wise_multiply"),
                        "r_product"),
            1)

        self._f_prob_h = f_prob_h = self.w * htr + self.b
        self._f_score_h = f_score_h = tf.square(tf.subtract(f_prob_h, self._A_w))

        self._f_prob_hn = f_prob_hn = self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_hn_batch, tf.multiply(self._neg_hn_con_batch, self._neg_t_con_batch)), 2)
        ) + self.b
        self._f_score_hn = f_score_hn = tf.reduce_mean(tf.square(f_prob_hn), 1)

        self._f_prob_tn = f_prob_tn = self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_tn_batch, tf.multiply(self._neg_h_con_batch, self._neg_tn_con_batch)), 2)
        ) + self.b
        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(f_prob_tn), 1)

        self.this_loss = this_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size

        # L2 regularizer
        self._regularizer = regularizer = tf.add(tf.add(tf.divide(tf.nn.l2_loss(self._h_batch), self.batch_size),
                                                        tf.divide(tf.nn.l2_loss(self._t_batch), self.batch_size)),
                                                 tf.divide(tf.nn.l2_loss(self._r_batch), self.batch_size))

        self.main_loss = tf.add(this_loss, self.reg_scale * regularizer)

    # override
    def define_psl_loss(self):
        self.psl_prob = self.w * tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1) + self.b
        self.compute_psl_loss()  # in tf_parts