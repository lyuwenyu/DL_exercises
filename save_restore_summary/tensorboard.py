
import tensorflow as tf 
import numpy as np 

class DataLoader():
	def __init__(self, n=2000, batch_size=3):
		
		#self.x = np.reshape(np.linspace(1, 20, n), (-1, 1))
		self.x = np.random.randn(n,1) *10
		self.y = self.x * 10 + 2.0 

		self.n = n
		self.batch_size = batch_size
		self._cur = 0
		self._perm = np.random.permutation(self.n)

	def next_batch(self):
		
		if self._cur+self.batch_size >= self.n:
			self._cur = 0
			self._perm = np.random.permutation(self.n)

		x = self.x[self._cur: self._cur+ self.batch_size, :]
		y = self.y[self._cur: self._cur+ self.batch_size, :]
		self._cur = self._cur + self.batch_size
		#print x, y
		return x, y


#with tf.name_scope('Input'):
#	x = tf.placeholder(dtype=tf.float32, shape=[None, 1])
#	y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
#
#with tf.variable_scope('wxpb') :
#	W = tf.get_variable(name='W', shape=[1,1], initializer=tf.truncated_normal_initializer(stddev=0.1))
#	b = tf.get_variable(name='b', shape=[1], initializer=tf.constant_initializer(value=0.0))
#	pre =  tf.matmul(x, W)+ b
#
#with tf.name_scope('Loss'):
#	loss = tf.reduce_mean(tf.square( tf.sub(pre, y) ))
#
#with tf.name_scope('Optimizer'):
#
#	opt = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
#
#with tf.name_scope('Summary') :
#	tf.histogram_summary('w', W)
#	tf.histogram_summary('b', b)
#	tf.scalar_summary('loss', loss)
#	merged_summaries = tf.merge_all_summaries()

with tf.name_scope('input_data'):
	x = tf.placeholder(dtype=tf.float32, shape=[None, 1])
	y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

with tf.name_scope('model'):
	W = tf.get_variable(name='W', shape=[1,1], initializer=tf.truncated_normal_initializer(stddev=0.1))
	b = tf.get_variable(name='b', shape=[1], initializer=tf.constant_initializer(value=0.0))
	pre =  tf.matmul(x, W)+ b


loss = tf.reduce_mean(tf.square( pre- y ))

tf.add_to_collection("pre", pre)

opt = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

tf.summary.histogram('w', W)
#tf.histogram_summary('b', b)
tf.summary.scalar('loss', loss)

merged_summaries = tf.summary.merge_all()

saver = tf.train.Saver()
saver_v2 = tf.train.Saver({"w": W})
#train_writer = tf.summary.FileWriter('./logs'+'train', graph)

with tf.Session() as sess :

	#writer = tf.train.SummaryWriter('./logs', sess.graph)
	
	writer = tf.summary.FileWriter('./logs'+'/train', sess.graph)
	
	sess.run(tf.global_variables_initializer())

	#saver = tf.train.Saver()

	dataload = DataLoader()
	#sess.run( tf.initialize_all_variables())
	for step in xrange(5000):

		xi, yi = dataload.next_batch()

		feed_dict = {x: xi, y: yi }

		_, summ = sess.run([opt, merged_summaries], feed_dict= feed_dict)

		if step % 100 == 0:
			writer.add_summary(summ, step)
		
		if step % 2000 ==0:	
			saver.save(sess, './models/model', global_step=step)
			saver_v2.save(sess, './models/v2_model', global_step=step)
	print sess.run([W, b])
