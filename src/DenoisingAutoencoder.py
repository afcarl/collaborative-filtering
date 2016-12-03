# coding=utf-8
import tensorflow as tf
import numpy as np
from Utils import xavier_init


class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(),
                 scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),
                                                     self.weights['w1']),
                                           self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # cost

        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.sub(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X,
                                                                          self.scale: self.training_scale
                                                                          })
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X,
                                                   self.scale: self.training_scale
                                                   })

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X,
                                                     self.scale: self.training_scale
                                                     })

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X,
                                                             self.scale: self.training_scale
                                                             })

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])


class MaskingNoiseAutoencoder(object):
    def __init__(self, session, n_input, n_hidden,
                 alpha=1.0, beta=0.7,
                 transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(),
                 dropout_probability=0.25,
                 weight_decay=0.02,
                 scale=0.2):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.dropout_probability = dropout_probability
        self.keep_prob = tf.placeholder(tf.float32)

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # Weight regularization
        l2_reg = tf.contrib.layers.l2_regularizer(weight_decay)
        for w in self.weights.values(): l2_reg(w)
        # Define model
        self.x_orig = tf.placeholder(tf.float32, [None, self.n_input], name='X')
        self.missing_values = tf.placeholder(tf.bool, [None, self.n_input], name='missing')
        zero_like = tf.zeros_like(self.x_orig)
        # Hide original missing values
        self.x = tf.select(self.missing_values, zero_like, self.x_orig)
        # Apply drop-out (confusion set) and save the dropout mask
        self.dropped_out = tf.nn.dropout(self.x, self.keep_prob)
        self.dropout_mask = tf.equal(self.x, self.dropped_out)
        self.hidden = self.transfer(tf.add(tf.matmul(self.dropped_out + scale * tf.random_normal((n_input,)),
                                                     self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        '''
        Before backpropagation, unknown ratings are turned
        to zero error, prediction errors are reweighed by α and
        reconstruction errors are reweighed by β
        '''
        error = tf.sub(self.reconstruction, self.x)
        error = tf.select(self.missing_values, zero_like, error)
        loss = tf.select(self.dropout_mask, alpha * error, beta * error)
        reweighted_loss = tf.select(tf.equal(self.x, zero_like), zero_like, loss)
        self.cost = 0.5 * tf.reduce_sum(tf.pow(reweighted_loss, 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = session
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X, missing_vals):
        assert X.dtype == np.float32
        assert missing_vals.dtype == np.bool
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict={self.x_orig: X,
                                             self.keep_prob: self.dropout_probability,
                                             self.missing_values: missing_vals})
        return cost

    def calc_total_cost(self, X, missing_vals):
        return self.sess.run(self.cost, feed_dict={self.x_orig: X,
                                                   self.missing_values : missing_vals,
                                                   self.keep_prob: 1.0})

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.keep_prob: 1.0})

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.keep_prob: 1.0})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])
