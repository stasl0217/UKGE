# definition of hyperparameters
# set by arguments of training

from os.path import join

verbose = False
whichdata = None
whichmodel = None
n_epoch = 100
learning_rate = 0.01
batch_size = 1024
val_save_freq = 10
dim = 128
neg_per_pos = 10  # Number of negative samples per (h,r,t). default 10.
p_neg = 1
p_psl = 0.2  # coefficient
n_psl = 1
prior_psl = 0
test_hrmap = False
reg_scale = None  # coefficient for regularizer (lambda)
data_dir = None
save_dir = None

def data_dir():
    return join('./data', whichdata)

def description():
    str = 'model:%s, batch size:%d, lr:%f, lambda:%f, dim:%d, n_neg:%f, p_neg:%f, p_psl:%f, n_psl:%d, prior_psl:%f' % (whichmodel,batch_size,
                                                                                         learning_rate, reg_scale, dim,
                                                                                         neg_per_pos, p_neg, p_psl,
                                                                                         n_psl, prior_psl)
    return str
