#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 10:51:10 2017

@author: zhouying
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
max_steps = 1000
learning_rate = 0.001
dropout = 0.9
data_dir = '/tmp/tensorflow/mnist/input_data'
log_dir = 'tmp/tensorflow/mnist/logs/mnist_with_summaries'
mnist = input_data.read_data_sets(data_dir,one_hot = True)
sess = tf.InteractiveSession()
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None,784],name='x-input')
