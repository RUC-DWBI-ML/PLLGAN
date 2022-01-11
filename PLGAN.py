import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow.contrib.layers as tcl
from tqdm import tqdm
from dataset_load import pml
import dataset_load as du


maxepoch = 501
batch_size = 64

#data = cifar
z_dim = 100
y_dim = du.m_categories  # condition
X_dim = du.p_features



def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


# for test
def sample_y(m, n, ind):
    y = np.zeros([m, n])
    for i in range(m):
        y[i, ind] = 1
    return y


def concat(z, y):
    return tf.concat([z, y], 1)


def get_batch(x, y, batch_size):
    index = np.random.choice(x.shape[0], batch_size)
    batch_x = x[index, :]
    batch_y = y[index, :]
    return batch_x, batch_y


def acc(probvec, objvec, verbose=True):
    total = probvec.shape[0]
    probvec = np.argmax(probvec, axis=1)
    probvec = probvec.reshape((total, 1))
    objvec = np.argmax(objvec, axis=1)
    objvec = objvec.reshape((total, 1))
    accuracy = np.sum(probvec == objvec) / total
    return accuracy


class G_mlp(object):
    def __init__(self):
        self.name = 'G_mlp'

    def __call__(self, z):
        with tf.variable_scope(self.name) as scope:
            g = tcl.fully_connected(z, du.h1_size, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            g = tcl.fully_connected(g, 64, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            g = tcl.fully_connected(g, 64, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            g = tcl.fully_connected(g, 64, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            g = tcl.fully_connected(g, X_dim, activation_fn=tf.nn.tanh, normalizer_fn=tcl.batch_norm)
            # g = tf.reshape(g, tf.stack([tf.shape(z)[0], 64, 64, 3]))
            return g

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class D_mlp(object):
    def __init__(self):
        self.name = "D_mlp"

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
             #adjusting according to data
            d = tcl.fully_connected(x, du.h1_size, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            d = tcl.fully_connected(d, du.h1_size, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            d = tcl.fully_connected(d, du.h1_size, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
 #           d = tcl.fully_connected(d, du.h1_size, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
  #          d = tcl.fully_connected(d, du.h1_size, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            d = tcl.fully_connected(d, du.h1_size, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            d = tcl.fully_connected(d, 64, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
#            d = tcl.fully_connected(d, 64, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            d=tf.layers.dropout(inputs=d, rate=0.7)
            logit = tcl.fully_connected(d, y_dim+1, activation_fn=tf.nn.softmax)
        return logit

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


# param
generator = G_mlp()
discriminator = D_mlp()

# placeholder
X = tf.placeholder(tf.float32, shape=[None, X_dim])
# X = tf.placeholder(tf.float32, shape=[None, size, size, channels])
y = tf.placeholder(tf.float32, shape=[None, y_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

# Generation
G_sample = generator(concat(z, y))
# Discrimination
# Step 1|(X, 0)
D_real_c = discriminator(concat(X, tf.zeros_like(y)))[:, :-1]
D_fake_c = discriminator(concat(G_sample, tf.zeros_like(y)), reuse=True)[:, :-1]
# Step 2|(X, y)
D_real_s = discriminator(concat(X, y), reuse=True)[:, -1:]
D_fake_s = discriminator(concat(G_sample, y), reuse=True)[:, -1:]

D_real = concat(D_real_c, D_real_s)
D_fake = concat(D_fake_c, D_fake_s)

D_real_act = tf.nn.softmax(D_real_c)

additional_dim = tf.ones_like(y)[:, :1]
D_real_target = concat(y, tf.zeros_like(additional_dim))  # (y, 0)
D_fake_target = concat(tf.zeros_like(y), additional_dim)  # (0, 1)

#loss function Least_square
R_loss = -tf.reduce_mean(tf.square(D_real - D_real_target))
F_loss = -tf.reduce_mean(tf.square(D_fake - D_fake_target))
# loss
D_loss = -(R_loss + F_loss)
G_loss = -F_loss


# solver
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=discriminator.vars)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=generator.vars)

for var in discriminator.vars:
    print(var.name)

acc_train_10 = np.zeros((maxepoch, 10))
acc_test_10 = np.zeros_like(acc_train_10)
dloss_table = np.zeros_like(acc_train_10)
dloss_mean_table = np.zeros((maxepoch, 1))
gloss_table = np.zeros_like(dloss_table)
gloss_mean_table = np.zeros_like(dloss_mean_table)

saver = tf.train.Saver()
sess = tf.Session()

for fold in range(10):
    print('No.%d Fold |' % (fold + 1))
    pml.traintest(fold)
    train_x = pml.train_x
    train_y = pml.train_y
    train_py = pml.train_py
    test_x = pml.test_x
    test_y = pml.test_y
    fig_count = 0

    sess.run(tf.global_variables_initializer())
    for epoch in tqdm(range(maxepoch)):
        # sample batch
        x_b, y_b = pml(batch_size)
        noise = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)

        # update D
        sess.run(D_solver, feed_dict={X: x_b, y: y_b, z: sample_z(batch_size, z_dim)})
        # update G
        k = 1
        for _ in range(k):
            sess.run(G_solver, feed_dict={y: y_b, z: sample_z(batch_size, z_dim)})

        # save img, model. print loss
        if epoch % 100 == 0:
            D_loss_curr = sess.run(D_loss,
                                   feed_dict={X: x_b, y: y_b, z: sample_z(batch_size, z_dim)})
            G_loss_curr = sess.run(G_loss,
                                   feed_dict={y: y_b, z: sample_z(batch_size, z_dim)})
            print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr))

        if epoch == maxepoch - 1:
            # Training Accuracy on all trainset
            train_outputs = sess.run(D_real_act, feed_dict={X: train_x, y: train_py})
            train_acc = acc(train_outputs, train_y, verbose=True)
            acc_train_10[epoch, fold] = train_acc
            # Testing Accuracy on all testset
            test_outputs = sess.run(D_real_act, feed_dict={X: test_x, y: test_y})
            test_acc = acc(test_outputs, test_y, verbose=True)
            acc_test_10[epoch, fold] = test_acc
            # accuracy_array[fig_count, 0] = accuracy
            print('Accuracy | Train: %.3f; Test: %.3f' % (train_acc, test_acc))
        if epoch % 100 == 0 and fold == 0:
            outputs = sess.run(D_real_act, feed_dict={X: test_x, y: test_y})

#evaluations.append(average)
#np.savetxt('./Evaluations/' + dataname + _r + '.csv', evaluations, delimiter=',')
