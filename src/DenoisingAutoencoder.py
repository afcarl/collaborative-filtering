# coding=utf-8
import tensorflow as tf
import numpy as np
from Utils import xavier_init


class DenoisingAutoencoder(object):
    def __init__(self, session, n_input, n_hidden,
                 alpha=1.0, beta=0.7,
                 transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(),
                 dropout_probability=0.25,
                 weight_decay=0.02,
                 noise_scale=0.1):
        """
        Define the tensorflow model for the hybrid autoencoder.
        :param session: tensorflow session
        :param n_input: number of input units
        :param n_hidden: number of hidden units for the first layer
        :param alpha: weight of the prediction error in the loss function
        :param beta: weight of the denoising error in the loss function
        :param transfer_function: typically use the hyperbolic tangent
        :param optimizer: optimizer to user (default: Adam optimizer)
        :param dropout_probability: hide ratio
        :param weight_decay: regularization factor
        :param noise_scale: variance for the normally distributed noise corrupting inputs
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.dropout_probability = dropout_probability
        self.keep_prob = tf.placeholder(tf.float32)

        self.scale = noise_scale
        self.noise_scale = tf.placeholder(tf.float32)

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # Apply weight regularization, regularization can help overfitting
        l2_reg = tf.contrib.layers.l2_regularizer(weight_decay)
        for w in self.weights.values(): l2_reg(w)
        # Define input and hidden layer
        self.x_orig = tf.placeholder(tf.float32, [None, self.n_input], name='X')
        self.missing_mask = tf.placeholder(tf.bool, [None, self.n_input], name='missing')
        zero_like = tf.zeros_like(self.x_orig) # tensor
        # Hide original missing values
        self.x = tf.where(self.missing_mask, zero_like, self.x_orig)
        # Apply drop-out and save the dropout mask
        self.dropped_out = tf.nn.dropout(self.x, self.keep_prob)
        self.dropout_mask = tf.equal(self.x, self.dropped_out)
        # Y = tanh(X_corr * W1 + b1)
        self.hidden = self.transfer(tf.add(tf.matmul(self.dropped_out + self.noise_scale * tf.random_normal((n_input,)),
                                                     self.weights['w1']), self.weights['b1']))
        # Y_prime = ? (hidden_output * W2 + b2)
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])


        '''
        Before backpropagation, unknown ratings are turned
        to zero error, prediction errors are reweighed by α and
        reconstruction errors are reweighed by β
        '''
        error = tf.subtract(self.reconstruction, self.x)
        masked_error = tf.where(self.missing_mask, zero_like, error)
        reweighted_loss = tf.where(self.dropout_mask, alpha * masked_error, beta * masked_error)

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
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x_orig: X,
                                                                          self.keep_prob: self.dropout_probability,
                                                                          self.noise_scale: self.scale,
                                                                          self.missing_mask: missing_vals})
        return cost, opt

    def calc_total_cost(self, X, missing_vals):
        cost = self.sess.run(self.cost, feed_dict={self.x_orig: X,
                                                   self.missing_mask: missing_vals,
                                                   self.keep_prob: 1.0,
                                                   self.noise_scale: 0.0})
        return np.sqrt(2.0 * cost)

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.noise_scale: 0.0, self.keep_prob: 1.0})

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.noise_scale: 0.0, self.keep_prob: 1.0})

    def restore_model(self, path):
        saver = tf.train.Saver()
        vars = saver.restore(self.sess, path)
