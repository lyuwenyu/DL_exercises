
import tensorflow as tf 
from skimage import io, transform
import random
import numpy as np 
from collections import deque
import pong


#define hyperparameters
ACTIONS = 3
#learning rate
GAMMA = 0.99
#update our graident or training time
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
#how many frams to anneal epsilon
EXPLORE = 500000
OBSERVE = 50000
REPLAY_MEMORY = 50000
#batch size
BATCH = 100


#creat TF graph
def createGraph():

	#first convolutional layer, and bias vector
	W_conv1 = tf.Variable(tf.truncated_normal(shape=[8,8,4,32], stddev=0.1))
	b_conv1 = tf.Variable(tf.zeros([32]))

	#second
	W_conv2 = tf.Variable(tf.truncated_normal(shape=[4,4,32,64], stddev=0.1))
	b_conv2 = tf.Variable(tf.zeros([64]))

	#third
	W_conv3 = tf.Variable(tf.truncated_normal(shape=[3,3,64,64], stddev=0.1))
	b_conv3 = tf.Variable(tf.zeros([64]))	

	#four
	W_fc4 = tf.Variable(tf.truncated_normal(shape=[784, 784], stddev=0.1))
	b_fc4 = tf.Variable(tf.zeros([784]))

	#last
	W_fc5 = tf.Variable(tf.truncated_normal(shape=[784, ACTIONS], stddev=0.1))
	b_fc5 = tf.Variable(tf.zeros([ACTIONS]))

	#input for pixel data
	s = tf.placeholder(dtype=tf.float32, [None, 84, 84, 84])

	#compute RELU activation function
	#on 2d convolutions
	#given 4d inputes and filter tensors

	conv1 = tf.nn.relu( tf.nn.conv2d(s, W_conv1, strides=[1, 4,4, 1], padding='VALID')+ b_conv1)
	conv2 = tf.nn.relu( tf.nn.conv2d(conv1, W_conv2, strides=[1, 4,4, 1], padding='VALID')+ b_conv2)
	conv3 = tf.nn.relu( tf.nn.conv2d(conv2, W_conv3, strides=[1, 4,4, 1], padding='VALID')+ b_conv3)

	conv3_flat = tf.reshape(conv3, [-1, 3136])
	fc4 = tf.nn.relu( tf.matmul( conv3_flat, W_fc4) + b_fc4 )
	fc5 = tf.matmul( fc5, W_fc5) + b_fc5

	return s, fc5


def main():

	#create session
	sess = tf.InteractiveSession()
	#input layer and our output layer
	inp, out = createGraph()

	trainGraph(inp. out, sess)

if __name__ = '__main__':
	main()
