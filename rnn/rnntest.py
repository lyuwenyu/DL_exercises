import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 



BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10


def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS

    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]



class LSTMRNN(object):

	def __init__(self,n_steps, input_size, output_size, cell_size, batch_size):
		
		self.n_steps = n_steps
		self.input_size = input_size
		self.output_size = output_size
		self.cell_size = cell_size
		self.batch_size = batch_size

		with tf.name_scope('inputs'):
			self.xs = tf.placeholder(tf.float32, [None, self.n_steps, self.input_size], name='xs')
			self.ys = tf.placeholder(tf.float32, [None, self.n_steps, self.output_size], name='ys')

		with tf.variable_scope('in_hidden'):
			self.add_input_layer()
			print 'add_input_layer'

		with tf.variable_scope('lstm_cell'):
			self.add_cell()
			print 'add_cell'

		with tf.variable_scope('out_hidden_'):
			self.add_out_layer()

		with tf.name_scope('cost'):
			self.comput_cost()

		with tf.name_scope('train'):
			global_step = tf.Variable(0, trainable=False)
			lr = tf.train.exponential_decay(0.01, global_step, 1000, 0.96, staircase=True)
			self.train_op = tf.train.AdamOptimizer(lr).minimize(self.cost, global_step=global_step)

	def add_input_layer(self):
		l_in_x = tf.reshape(self.xs, [-1, self.input_size])
		ws_in = self._weight_variable([self.input_size, self.cell_size])
		bs_in = self._bias_variable([self.cell_size])
		with tf.name_scope('wx_plus_b'):
			l_in_y = tf.matmul(l_in_x, ws_in) + bs_in
		self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size])
		

	def add_cell(self):
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size)
		self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
		
		#method-0 
		self.cell_final_state = self.cell_init_state
		self.cell_outputs = []
		with tf.variable_scope('lstm_cells') as scope:
			for t in range(self.n_steps):
				if t > 0 : 
					scope.reuse_variables()
				#here do not use cell_init_state [if cell_init_state is asigned in the session]
				cell_out, self.cell_final_state = lstm_cell(self.l_in_y[:, t, :], self.cell_final_state)
				self.cell_outputs.append(cell_out)
		
			self.cell_outputs = tf.transpose(self.cell_outputs, [1, 0, 2])
			print self.cell_outputs.get_shape()
		
		#method-1
		#self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(lstm_cell, self.l_in_y, initial_state=self.cell_init_state)
		
		#method-2
		#inputx = tf.reshape(tf.transpose(self.l_in_y, [1, 0, 2]), [-1, self.cell_size])
		#inputx = tf.split(0, self.n_steps, inputx)
		#cell_outputs, self.cell_final_state = tf.nn.rnn(lstm_cell, inputx, self.cell_init_state)
		#self.cell_outputs = tf.transpose(cell_outputs, [1,0,2])

	def add_out_layer(self):
		l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size])
		ws_out = self._weight_variable([self.cell_size, self.output_size])
		bs_out = self._bias_variable([self.output_size])
		with tf.name_scope('wx_plus_b'):
			self.pred = tf.matmul(l_out_x, ws_out) + bs_out

	def comput_cost(self):
		losses = tf.nn.seq2seq.sequence_loss_by_example([tf.reshape(self.pred, [-1])], [tf.reshape(self.ys, [-1])], [tf.ones([self.batch_size*self.n_steps], dtype=tf.float32)], average_across_timesteps=True, softmax_loss_function=self.ms_error)
		with tf.name_scope('ave_loss'):
			self.cost = tf.reduce_mean(losses)
			tf.scalar_summary('cost', self.cost)

	def ms_error(self, y_pre, y_target):
		return tf.square(tf.sub(y_pre, y_target))

	def _weight_variable(self, shape, name='weights'):
		init = tf.truncated_normal_initializer(stddev=0.1)
		return tf.get_variable(name=name, shape=shape, initializer=init)

	def _bias_variable(self, shape, name='bias'):
		init = tf.constant_initializer(0.)
		return tf.get_variable(name=name, shape=shape, initializer=init)

		
if __name__ == '__main__':

	model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
	with tf.Session() as sess:

		merged = tf.merge_all_summaries()
		writer = tf.train.SummaryWriter('logs', sess.graph)

		sess.run( tf.initialize_all_variables())

		plt.ion()
		plt.show()

		for i in range(1000) :
			seq, res, xs = get_batch()

			if i==0:
				feed_dict = {model.xs: seq, model.ys: res}
			else:
				#feed_dict = {model.xs: seq, model.ys: res}
				feed_dict = {model.xs: seq, model.ys: res, model.cell_init_state: state}

			_, pred, state = sess.run([model.train_op, model.pred, model.cell_final_state], feed_dict=feed_dict)

			plt.plot(xs[0, :], res[0].flatten(), 'r')
			plt.plot(xs[0, :], pred.flatten()[:TIME_STEPS], 'b')

			plt.ylim((-1.2, 1.2))
			plt.draw()
			plt.pause(0.2)

			if i%300 == 0:
				print i, 
				result = sess.run(merged, feed_dict)
				writer.add_summary(result, i)
				#plt.ylim((-1.2, 1.2))
				#plt.draw()
				#plt.pause(0.3)
				#sys.stdout.flush()