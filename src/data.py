"""Processing of data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pickle
import pandas as pd
from os.path import join

from src import param


class Data(object):
    '''The abustrct class that defines interfaces for holding all data.
    '''

    def __init__(self):
        # concept vocab
        self.cons = []
        # rel vocab
        self.rels = []
        # transitive rels vocab
        self.index_cons = {}  # {string: index}
        self.index_rels = {}  # {string: index}
        # save triples as array of indices
        self.triples = np.array([0])  # training dataset
        self.val_triples = np.array([0])  # validation dataset
        self.soft_logic_triples = np.array([0])

        # (h,r,t) tuples(int), no w
        # set containing train, val, test (for negative sampling).
        self.triples_record = set([])
        self.weights = np.array([0])

        self.neg_triples = np.array([0])
        # map for sigma
        # head per tail and tail per head (for each relation). used for bernoulli negative sampling
        self.hpt = np.array([0])
        self.tph = np.array([0])
        # recorded for tf_parts
        self.dim = 64
        self.batch_size = 1024
        self.L1 = False

    def load_triples(self, filename, splitter='\t', line_end='\n'):
        '''Load the dataset'''
        triples = []
        last_c = -1
        last_r = -1
        hr_map = {}
        tr_map = {}

        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            if self.index_cons.get(line[0]) == None:
                self.cons.append(line[0])
                last_c += 1
                self.index_cons[line[0]] = last_c
            if self.index_cons.get(line[2]) == None:
                self.cons.append(line[2])
                last_c += 1
                self.index_cons[line[2]] = last_c
            if self.index_rels.get(line[1]) == None:
                self.rels.append(line[1])
                last_r += 1
                self.index_rels[line[1]] = last_r
            h = self.index_cons[line[0]]
            r = self.index_rels[line[1]]
            t = self.index_cons[line[2]]
            w = float(line[3])

            triples.append([h, r, t, w])
            self.triples_record.add((h, r, t))
        return np.array(triples)

    def load_data(self, file_train, file_val, file_psl=None, splitter='\t', line_end='\n'):

        self.triples = self.load_triples(file_train, splitter, line_end)
        self.val_triples = self.load_triples(file_val, splitter, line_end)

        if file_psl is not None:
            self.soft_logic_triples = self.load_triples(file_psl, splitter, line_end)

        # calculate tph and hpt
        tph_array = np.zeros((len(self.rels), len(self.cons)))
        hpt_array = np.zeros((len(self.rels), len(self.cons)))
        for h_, r_, t_, w in self.triples:  # only training data
            h, r, t = int(h_), int(r_), int(t_)
            tph_array[r][h] += 1.
            hpt_array[r][t] += 1.
        self.tph = np.mean(tph_array, axis=1)
        self.hpt = np.mean(hpt_array, axis=1)
        # print("-- total number of entities:", len(self.cons))

    # add more triples to self.triples_record to 'filt' negative sampling
    def record_more_data(self, filename, splitter='\t', line_end='\n'):
        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            if len(line) < 3:
                continue
            h = self.con_str2index(line[0])
            r = self.rel_str2index(line[1])
            t = self.con_str2index(line[2])
            w = line[3]
            if h != None and r != None and t != None:
                self.triples_record.add((h, r, t))
        # print("Loaded %s to triples_record." % (filename))
        # print("Update: total number of triples in set:", len(self.triples_record))

    def num_cons(self):
        '''Returns number of ontologies.

        This means all ontologies have index that 0 <= index < num_onto().
        '''
        return len(self.cons)

    def num_rels(self):
        '''Returns number of relations.

        This means all relations have index that 0 <= index < num_rels().
        Note that we consider *ALL* relations, e.g. $R_O$, $R_h$ and $R_{tr}$.
        '''
        return len(self.rels)

    def rel_str2index(self, rel_str):
        '''For relation `rel_str` in string, returns its index.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.index_rels.get(rel_str)

    def rel_index2str(self, rel_index):
        '''For relation `rel_index` in int, returns its string.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.rels[rel_index]

    def con_str2index(self, con_str):
        '''For ontology `con_str` in string, returns its index.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.index_cons.get(con_str)

    def con_index2str(self, con_index):
        '''For ontology `con_index` in int, returns its string.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.cons[con_index]

    def rel(self):
        return np.array(range(self.num_rels()))

    def corrupt_pos(self, triple, pos):
        """
        :param triple: [h, r, t]
        :param pos: index position to replace (0 for h, 2 fot t)
        :return: [h', r, t] or [h, r, t']
        """
        hit = True
        res = None
        while hit:
            res = np.copy(triple)
            samp = np.random.randint(self.num_cons())
            while samp == triple[pos]:
                samp = np.random.randint(self.num_cons())
            res[pos] = samp
            # # debug
            # if tuple(res) in self.triples_record:
            #     print('negative sampling: rechoose')
            #     print(res)
            if tuple(res) not in self.triples_record:
                hit = False
        return res

    # bernoulli negative sampling
    def corrupt(self, triple, neg_per_positive, tar=None):
        """
        :param triple: [h r t]
        :param tar: 't' or 'h'
        :return: np.array [[h,r,t1],[h,r,t2],...]
        """
        # print("array.shape:", res.shape)
        if tar == 't':
            position = 2
        elif tar == 'h':
            position = 0
        res = [self.corrupt_pos(triple, position) for i in range(neg_per_positive)]
        return np.array(res)

    class index_dist:
        def __init__(self, index, dist):
            self.dist = dist
            self.index = index
            return

        def __lt__(self, other):
            return self.dist > other.dist

    # bernoulli negative sampling on a batch
    def corrupt_batch(self, t_batch, neg_per_positive, tar=None):
        res = np.array([self.corrupt(triple, neg_per_positive, tar) for triple in t_batch])
        return res

    def save(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        # print("Save data object as", filename)

    def load(self, filename):
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        print("Loaded data object from", filename)

    def save_meta_table(self, save_dir):
        """
        save index-con, index-rel table to file.
        File: idx_concept.csv, idx_relation.csv
        :return:
        """
        idx_con_path = join(save_dir, 'idx_concept.csv')
        df_con = pd.DataFrame({'index': list(self.index_cons.values()), 'concepts': list(self.index_cons.keys())})
        df_con.sort_values(by='index').to_csv(idx_con_path, index=None)

        idx_rel_path = join(save_dir, 'idx_relation.csv')
        df_rel = pd.DataFrame({'index': list(self.index_rels.values()), 'relations': list(self.index_rels.keys())})
        df_rel.sort_values(by='index').to_csv(idx_rel_path, index=None)


class BatchLoader():
    def __init__(self, data_obj, batch_size, neg_per_positive):
        self.this_data = data_obj  # Data() object
        self.shuffle = True
        self.batch_size = batch_size
        self.neg_per_positive = neg_per_positive
        self.n_soft_samples = param.n_psl  # number of samples per batch

    def gen_psl_samples(self):
        # samples from probabilistic soft logic
        softlogics = self.this_data.soft_logic_triples  # [[A, B, base]]
        triple_indices = np.random.randint(0, softlogics.shape[0], size=self.n_soft_samples)
        samples = softlogics[triple_indices,:]
        soft_h, soft_r, soft_t, soft_lb = samples[:,0].astype(int),samples[:,1].astype(int),samples[:,2].astype(int), samples[:,3]
        soft_sample_batch = (soft_h, soft_r, soft_t, soft_lb)
        return soft_sample_batch



    def gen_batch(self, forever=False, shuffle=True, negsampler=None):
        """
        :param ht_embedding: for kNN negative sampling
        :return:
        """
        l = self.this_data.triples.shape[0]
        while True:
            triples = self.this_data.triples  # np.float64 [[h,r,t,w]]
            if shuffle:
                np.random.shuffle(triples)
            for i in range(0, l, self.batch_size):
                batch = triples[i: i + self.batch_size, :]
                if batch.shape[0] < self.batch_size:
                    batch = np.concatenate((batch, self.this_data.triples[:self.batch_size - batch.shape[0]]),
                                           axis=0)
                    assert batch.shape[0] == self.batch_size

                h_batch, r_batch, t_batch, w_batch = batch[:, 0].astype(int), batch[:, 1].astype(int), batch[:,
                                                                                                       2].astype(
                    int), batch[:, 3]
                hrt_batch = batch[:, 0:3].astype(int)


                if negsampler is None:
                    # all_neg_hn_batch = self.corrupt_batch(hrt_batch, self.neg_per_positive, "h")
                    # all_neg_tn_batch = self.corrupt_batch(hrt_batch, self.neg_per_positive, "t")

                    neg_hn_batch, neg_rel_hn_batch, \
                    neg_t_batch, neg_h_batch, \
                    neg_rel_tn_batch, neg_tn_batch \
                        = self.corrupt_batch(h_batch, r_batch, t_batch)
                else:
                    # neg_per_positive controlled by sampler
                    all_neg_hn_batch = negsampler.knn_negative_batch(hrt_batch, "h")
                    all_neg_tn_batch = negsampler.knn_negative_batch(hrt_batch, "t")

                yield h_batch.astype(np.int64), r_batch.astype(np.int64), t_batch.astype(
                    np.int64), w_batch.astype(
                    np.float32), \
                      neg_hn_batch.astype(np.int64), neg_rel_hn_batch.astype(np.int64), \
                      neg_t_batch.astype(np.int64), neg_h_batch.astype(np.int64), \
                      neg_rel_tn_batch.astype(np.int64), neg_tn_batch.astype(np.int64)
            if not forever:
                break

    def corrupt_batch(self, h_batch, r_batch, t_batch):
        N = self.this_data.num_cons()  # number of entities

        neg_hn_batch = np.random.randint(0, N, size=(
        self.batch_size, self.neg_per_positive))  # random index without filtering
        neg_rel_hn_batch = np.tile(r_batch, (self.neg_per_positive, 1)).transpose()  # copy
        neg_t_batch = np.tile(t_batch, (self.neg_per_positive, 1)).transpose()

        neg_h_batch = np.tile(h_batch, (self.neg_per_positive, 1)).transpose()
        neg_rel_tn_batch = neg_rel_hn_batch
        neg_tn_batch = np.random.randint(0, N, size=(self.batch_size, self.neg_per_positive))

        return neg_hn_batch, neg_rel_hn_batch, neg_t_batch, neg_h_batch, neg_rel_tn_batch, neg_tn_batch
