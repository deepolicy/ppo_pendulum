import tensorflow as tf
import numpy as np

def value_nn(state, obs_dim):

    with tf.variable_scope('value'):

        ly1 = tf.layers.dense(state, 64, activation=tf.nn.relu, name='ly1')
        ly2 = tf.layers.dense(ly1, 64, activation=tf.nn.relu, name='ly2')
        ly3 = tf.layers.dense(ly2, 1, activation=None, name='ly3')
        out = tf.reshape(ly3, [-1])
        return out
