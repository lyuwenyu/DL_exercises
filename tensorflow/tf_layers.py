import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

learning_rate = 0.001
training_iters = 1000
batch_size = 32

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
phase = tf.placeholder(tf.bool) #training

x1 = tf.reshape(x, [-1, 28,28,1])

conv1_relu = tf.layers.conv2d(x1, filters=32, kernel_size=[5,5], strides=[1,1], activation=tf.nn.relu)
#conv1_bn = tf.layers.batch_normalization(conv1, training=phase)
#conv1_relu = tf.nn.relu(conv1)

pool1 = tf.layers.max_pooling2d(conv1_relu, pool_size=[2,2], strides=[2,2], padding='same')

conv2_relu = tf.layers.conv2d(pool1, filters=64, kernel_size=[5,5], strides=[1,1], activation=tf.nn.relu)
#conv2_bn = tf.layers.batch_normalization(conv2, training=phase)
#conv2_relu = tf.nn.relu(conv2)

pool2 = tf.layers.max_pooling2d(conv2_relu, pool_size=[2,2], strides=[2,2], padding='same')

#print pool2.get_shape().as_list()
pool2 = tf.reshape(pool2, [-1, 4*4*64])
fc1 = tf.layers.dense(pool2, units= 1024)

fc1_doupout = tf.layers.dropout(fc1, training=phase, rate=0.3)

#out_w = tf.Variable(tf.random_normal([1024, 10]))
#out_b = tf.Variable(tf.random_normal([10]))
#out = tf.matmul( fc1_doupout , out_w) + out_b
pred = tf.nn.softmax(out)

cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
opt = tf.train.AdamOptimizer(0.001).minimize(cost)

corr = tf.equal( tf.argmax(pred, 1), tf.argmax(y, 1))
acc = tf.reduce_mean( tf.cast(corr, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    sess.run(init)
    
    for i in xrange(1000):
        xs,ys = mnist.train.next_batch(32)
        sess.run([opt], feed_dict={ x:xs, y:ys, phase: True})
        
        if i %100 ==0:
            accuracy, loss = sess.run([acc, cost], feed_dict={ x:xs, y:ys, phase: False})
            print loss, accuracy
    
    print sess.run(acc, feed_dict= {x: mnist.test.images[:100], y: mnist.test.labels[:100], phase: False})
    
    print sess.run([out, pred], feed_dict= {x: mnist.test.images[:1], y: mnist.test.labels[:1], phase: False})
