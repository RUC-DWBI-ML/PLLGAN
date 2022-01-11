import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import scale
##############################################################################
# Preset parameters
h1_size = 256  # the number of hidden layers' neurons
rate = 0.5  # the rate of dropping out
g_lr = 0.001  # generator learning rate
d_lr = 0.001  # discriminator learning rate
interval = 1  # output interval


# Preparation for data
dataname = 'MSRCv2'
caps = dataname.capitalize()
############################
data = loadmat('./'+dataname+'.mat')
############################
# Division of 10 folds
divide_file = './'+caps+'_indices.mat'
indices_dict = loadmat(divide_file)  # division file
indices = indices_dict['indices']  # the method of division
label = np.transpose(np.array(data['target'].todense()).astype(np.float32))#lostYahooNews/MSRCv2
plabel = np.transpose(np.array(data['partial_target'].todense()).astype(np.float32))

############################
# Data processing
features = data['data']  # feature space
features_scl = scale(features, axis=0)  # scale

# Data structure
data_shape = features.shape
n_cases = data_shape[0]  # the number of cases
p_features = data_shape[1]  # the number of features
m_categories = label.shape[1]  # the number of categories


############################
# processing of partial label
def pl_process(pl):
    pl_reg = np.zeros(shape=pl.shape).astype(np.float32)
    pl_num = np.zeros(shape=[n_cases, 1])
    for i in range(n_cases):
        pl_num[i, ] = np.sum(pl, axis=1)[i]
        pl_reg[i, :] = pl[i, :]/pl_num[i, ]

    num_l = np.zeros(shape=[data_shape[0], 1])
    for i in range(0, data_shape[0]):
        num_l[i] = np.where(label[i, :] == 1)[0][0]
    return pl_reg, pl_num, num_l



plabel_reg, plabel_num, num_label = pl_process(plabel)


def get_batch(x, y, batch_size, net='MLP'):
    index = np.random.choice(x.shape[0], batch_size)
    if net == 'MLP':
        batch_x = x[index, :]
    else:
        batch_x = x[index, :, :, :]
    batch_y = y[index, :]
    return batch_x, batch_y


class Complete(list):
    def __init__(self, completedata):
        super().__init__()
        self.features = completedata[0]
        self.labels = completedata[1]
        self.plabels = completedata[2]
        self.indices = completedata[3]
        self.complete = np.hstack((self.features, self.labels, self.plabels, self.indices))
        self.shape = self.features.shape
        self.ndim = self.shape[0]
        self.mdim = self.shape[1]
        self.cats = self.labels.shape[1]
        self.folds = 10

    def traintest(self, kfold=0):
        a = np.where(self.indices != (kfold+1))[0]
        b = np.where(self.indices == (kfold+1))[0]
        self.train_indices = a
        self.test_indices = b
        self.train_x = self.features[a, :]
        self.train_y = self.labels[a, :]
        self.train_py = self.plabels[a, :]

        self.test_x = self.features[b, :]
        self.test_y = self.labels[b, :]
        self.trainsize = len(a)
        self.testsize = len(b)

    def __call__(self, batch_size):
        batch_x, batch_y = get_batch(self.train_x, self.train_py, batch_size)
        return batch_x, batch_y


complete = [features_scl, label, plabel, indices]
pml = Complete(complete)
