import tensorflow as tf 
import numpy as np 
import csv
import matplotlib.pyplot as plt 


class data_loader():

	def __init__(self, data_file='./data/data/eth/univ/pixel_pos.csv'):
		
		self._cur = 0
		self.data_file = data_file
		self.init_data()

	def init_data(self):

		with open(self.data_file, 'rb') as f:
			raw_data = csv.reader(f)
			for j in raw_data:
				ll= np.array(j).shape[0] # 5492 6544
				break
			data = np.zeros((4, ll))

		with open(self.data_file, 'rb') as f:

			raw_data = csv.reader(f)
			for i, line in enumerate(raw_data):
				data[i, :] = line

			pers = np.unique(data[1, :])
			self.train_data = {}
			self.person = []

			for per in pers:
				if data[2:4, data[1,:] == per].shape[1] > 6:
					self.train_data[per] = data[2:4, data[1,:] == per]
					self.person.append(per)

			self._perm = np.random.permutation(np.arange(len(self.person)))

	def load_next_person(self):

		if self._cur >= len(self.person):
			self._cur = 0
			self._perm = np.random.permutation(np.arange(len(self.person)))

		xs = self.train_data[ self.person[self._perm[self._cur]] ].transpose()[:-1].reshape(1, -1, 2)
		ys = self.train_data[ self.person[self._perm[self._cur]] ].transpose()[1: ].reshape(1, -1, 2)

		self._cur +=1

		return xs, ys

	def test_data(self, start=100, memory_step=10, seq_len=10):
		
		for per in self.person[start:]:

			seq_lenx = self.train_data[per].shape[1]
			#print seq_len
			try :
				if seq_lenx == seq_len:
					xs = self.train_data[per].transpose()[0:memory_step, :].reshape(1, -1, 2)
					ys = self.train_data[per].transpose()[memory_step:seq_len, :].reshape(1, -1, 2)
					return xs, ys
			except:
				if seq_lenx == seq_len-3:
					xs = self.train_data[per].transpose()[0:memory_step, :].reshape(1, -1, 2)
					ys = self.train_data[per].transpose()[memory_step:seq_len, :].reshape(1, -1, 2)
					return xs, ys


def linear(x, input_dim=2, out_dim=128, scope=None):
	with tf.variable_scope( scope or 'linear'):
		w = tf.get_variable(name='w', shape=[input_dim, out_dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
		b = tf.get_variable(name='b', shape=[out_dim], initializer=tf.constant_initializer(0.1))
		return tf.matmul(x, w)+b

def sample_gaussian(corr):
	mux, muy, sx, sy, rho = corr
	mean = [mux, muy]
	cov = [[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]]
	x = np.random.multivariate_normal(mean, cov, 1)
	return x[0][0], x[0][1]


class social_lstm():

	def __init__(self, hid_dim=128, batch_size=1, n_steps=10, base_lr=0.001, stepsize=5000):

		self.hid_dim = hid_dim
		self.batch_size = batch_size
		self.n_steps = n_steps
		self.stepsize = stepsize
		self.base_lr = base_lr

		self.model()


	def model(self):

		with tf.name_scope('input'):
			self.xs = tf.placeholder(tf.float32, shape=[self.batch_size, None, 2])
			self.ys = tf.placeholder(tf.float32, shape=[self.batch_size, None, 2])

		with tf.variable_scope('embed_input'):
			inputx = tf.reshape(self.xs, [-1, 2])
			embed_input = linear(inputx, 2, 128)
			embed_input = tf.reshape(embed_input, [self.batch_size, -1, 128])

		with tf.variable_scope('lstm'):

			cell = tf.nn.rnn_cell.BasicLSTMCell(self.hid_dim)
			self.init_state = cell.zero_state(self.batch_size, dtype=tf.float32)

			self.outs, self.state = tf.nn.dynamic_rnn(cell, inputs=embed_input, \
				initial_state=self.init_state, dtype=tf.float32)

		with tf.variable_scope('output'):
			outputs = tf.reshape(self.outs, [-1, 128])
			outputs = linear(outputs, 128, 5)

			self.out_var = self.get_coef(outputs) #[o_mux, o_muy, o_sx, o_sy, o_corr]


		with tf.variable_scope('loss'):
			
			ys = tf.reshape(self.ys, [-1, 2])

			self.loss = self.loss_fun(self.out_var, ys)


		with tf.variable_scope('optimizer'):
			
			self.global_step = tf.Variable(0, trainable=False)
			self.lr = tf.train.exponential_decay(self.base_lr, self.global_step, self.stepsize, 0.1, staircase=True)

			self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
			tf.add_to_collection('train_op', self.opt)
			tf.add_to_collection('global_step', self.global_step)

		with tf.name_scope('summary'):
			
			tf.scalar_summary('loss', self.loss)
			self.summary = tf.merge_all_summaries()

		self.init = tf.initialize_all_variables()



	def loss_fun(self, out_res, target):

		p = self.tf_2d_nornal(out_res, target)
		eps = 1e-15
		res = -tf.log( tf.maximum(p, eps))
		return tf.reduce_sum(res)

	def tf_2d_nornal(self, out_var, target):

		mux, muy, sx, sy, rho = out_var
		x, y = tf.split(1, 2, target)

		normx = tf.sub(x , mux)
		normy = tf.sub(y , muy)
		sxsy = tf.mul(sx, sy)
		z = tf.square(tf.div(normx, sx)) + tf.square(tf.div(normy, sy)) - 2*tf.div(tf.mul(rho, tf.mul(normx, normy)), sxsy)
		negRho = 1 - tf.square(rho)
		result = tf.exp(tf.div(-z, 2*negRho))
		denom = 2 * np.pi * tf.mul(sxsy, tf.sqrt(negRho))
		result = tf.div(result, denom)

		return result


	def get_coef(self, output):

		z = tf.reshape(output, [-1, 5])
		z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(1, 5, z)
		z_sx, z_sy = tf.exp(z_sx), tf.exp(z_sy)
		z_corr = tf.tanh(z_corr)

		return z_mux, z_muy, z_sx, z_sy, z_corr





def train(trainx=False):

	data_load = data_loader()
	social = social_lstm()

	saver = tf.train.Saver()

	with tf.Session() as sess:
		
		writer = tf.train.SummaryWriter('./logs', sess.graph)
		#saver.restore(sess, tf.train.latest_checkpoint('./'))
		if trainx:

			#sess.run(social.init)
			#print sess.run(  social.global_step)
			saver.restore(sess, 'models-46000')
			print sess.run(  social.global_step)
			print sess.run( tf.assign( social.global_step, 0))

			for step in xrange(50000):

				xs, ys = data_load.load_next_person()

				if step ==0:
					feed_dict = {social.xs: xs, social.ys: ys}
				else:
					feed_dict = {social.xs: xs, social.ys: ys}

				_, loss, summ = sess.run([social.opt, social.loss, social.summary], feed_dict=feed_dict)

				if step % 200 == 0:
					global_step, lr = sess.run([social.global_step, social.lr], feed_dict=feed_dict)
					writer.add_summary(summ, step)
					print 'step: {} , lr: {}, lost: {}'.format(global_step, lr, loss)

				if step % 1000 == 0 and step!= 0 :
					saver.save(sess, './models', global_step=step)

		else:

			xs, ys = data_load.test_data()

			new_saver = tf.train.import_meta_graph('model-8000.meta')
			new_saver.restore(sess, tf.train.latest_checkpoint('./'))
			#saver.restore(sess, 'model')

			feed_dict = {social.xs: xs}

			out_var, state = sess.run([social.out_var, social.state], feed_dict=feed_dict)

			print out_var[0].shape


def restore_model():

	with tf.Session() as sess:

		##method-0
		saver = tf.train.import_meta_graph('models-45000.meta')
		ckpt = tf.train.get_checkpoint_state('./')
		saver.restore(sess, ckpt.model_checkpoint_path) #'models-46000'

		#res = tf.get_collection('global_step')[0]
		#print sess.run( res )
		#sess.run( tf.assign(res, 0) )
		#print sess.run( res )

		#varsx = tf.trainable_variables()
		#print type(varsx), len(varsx)
		#print sess.run(varsx[0])

		for var in tf.trainable_variables():
			print var.name 



		##method-1
		#saver = tf.train.Saver() # default graph
		#saver.restore(sess, 'models-46000')



def sample(step =10):
	
	data_load = data_loader()
	social = social_lstm()
	saver = tf.train.Saver()

	background = plt.imread('./bg.png')
	h, w = background.shape[0], background.shape[1]

	with tf.Session() as sess:
		
		#new_saver = tf.train.import_meta_graph('models-15000.meta')
		saver.restore(sess, tf.train.latest_checkpoint('./'))
		
		for lenx in range(20, 35):

			xs, ys = data_load.test_data(start=100, memory_step=15, seq_len=lenx)
			print xs.shape, ys.shape

			feed_dict = { social.xs: xs[:,0:-1, :] } 
			state = sess.run([social.state], feed_dict=feed_dict)
			
			res = []
			datai = np.expand_dims(xs[:,-1, :], 1)

			plt.ion()
			#plt.show()
			plt.imshow(background)

			for i in xrange(step):

				feed_dict = { social.xs: datai, social.init_state: state}

				state, out_var = sess.run([social.state, social.out_var], feed_dict = feed_dict)
				#print state

				resi = sample_gaussian(  np.squeeze(np.array(out_var)) )
				datai = np.array(resi).reshape(1, 1, 2)
				res.append(resi)

				plt.plot(resi[0]*h, resi[1]*w, 'ro')
				if i < ys.shape[1]:
					plt.plot(ys[0,i,0]*h, ys[0,i,1]*w, 'wo')

				#print ys[0,i,0], ys[0,i,1], resi[0], resi[1]

				plt.pause(1)
				plt.ylim(0, h)
				plt.xlim(0, w)

		#fig = plt.gcf()
		#fig.savefig('test.png')
		#res = np.array(res)
		
		#plt.plot(ys[0, :, 0], ys[0, :, 1])
		#print ys
		#annotate = [str(i) for i in range(step)]
		#plt.plot(res[:, 0], res[:, 1], 'ro')
		#plt.annotate()
		#plt.show()

if __name__ == '__main__':
	#train(trainx=True)
	#sample(step=10)
	restore_model()