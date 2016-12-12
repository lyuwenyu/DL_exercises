import tensorflow as tf 
import numpy as np 
import math
import glob
from skimage import io, transform

def lrelu(x, leak=0.2, name='lrelu'):
	
	with tf.variable_scope(name):

		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)

		return f1*x + f2*abs(x)

def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name='conv2d'):
	
	with tf.variable_scope(name):

		w = tf.get_variable(name='w', [k_h, k_w, input_.get_shape()[1], output_dim], \
			initializer=tf.truncated_normal_initializer(stddev=stddev))
		b = tf.get_variable(name='b', [output_dim], \
			initializer=tf.constant_initializer(0.0))
		
		return tf.nn.conv2d(input_, w, stride=[1, d_h, d_w, 1], padding='SAME')+ b

def conv2d_transpose(input_, output_shape, k_h, k_w, d_h, d_w, stddev=0.02, name='conv2d_transpose'):
	
	with tf.variable_scope(name):	

		w = tf.get_variable(name='w', shape=[k_h, k_w, output_shape[-1], input_.get_shape()[-1]], \
			initializer=tf.random_normal_initializer(stddev=stddev))
		b = tf.get_variable(name='b', shape=[output_shape[-1]], \
			initializer=tf.constant_initializer(0.0))
		
		return tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, stride=[1, d_h, d_w, 1])+b

def linear(input_, output_size, scope=None, stddev=0.02):	
	
	with tf.variable_scope(scope or 'linear'):

		shape = input_.get_shape().as_list()
		w = tf.get_variable(name='w', shape=[shape[1], output_size], \
			initializer=tf.random_normal_initializer(stddev=stddev))
		b = tf.get_variable(name='b', shape=[output_size], \
			initializer=tf.constant_initializer(0.0))
		
		return tf.matmul(input_, w) + b

#def bibary_cross_entropy(preds, targets, name=None):
#	eps = 1e-10
#	with tf.name_scope(name, default_name='cross_entropy', [preds, targets]):
#		preds = tf.convert_to_tensor(preds, name='preds')
#		targets = tf.convert_to_tensor(targets, name='targets')
#		return tf.reduce_mean(-(targets*tf.log(preds+eps))+ (1-targets)*tf.log(1.-preds+eps))


class batch_norm():
	
	def __init__(self):
		pass

	def __call__(self):
		pass


class data_loader():

	def __init__(self, batch_size, rangex=1.0, z_dim=10,img_h=64, img_w=64, imgs_dir='./'):

		self.batch_size = batch_size
		self.img_h = img_h
		self.img_w = img_w

		self.rangex = rangex
		self.z_dim = z_dim
		
		#self.imgs_dir = imgs_dir
		self.images_ = glob.glob('{}/*.jpg'.format(imgs_dir))
		self.perm_ = np.random.permutation(np.range(len(self.images_)))
		self.cur_ = 0

	def load_next_batch(self):

		imgs = self.image_loader()
		zs = self.z_generator()
		
		return imgs, zs

	def image_loader(self):

		if self.cur_ + self.batch_size >= len(self.images_):
			self.cur_ = 0
			self.perm_ = np.random.permutation(np.range(len(self.images_)))

		img_batch = np.zeros([self.batch_size, self.img_h, self.img_w, 3])

		for i, idx in enumerate(self.perm_[self.cur_: self.cur_+self.batch_size]):
			img = transform.resize(io.imread(self.images_[idx]), (self.img_h, self.img_w))
			img_batch[i, ...] = img

		self.cur_ += self.batch_size

		return img_batch

	def z_generator(self):

		z_batch = np.random.uniform(-self.rangex, self.rangex, [self.batch_size, self.z_dim])
		
		return z_batch.astype(np.float32)

## model
class DCGAN(object):

	def __init__(self, batch_size):

		self.batch_size = batch_size
		self.nb1 = batch_norm()

		self.model()

	def discriminator(self, img, reuse=False):

		if reuse:
			tf.get_variable_scope().reuse_variables()

		h0 = lrelu( conv2d(img, name='d_h0_conv') ) 
		h1 = lrelu( self.bn1( conv2d(h0, name='d_h1_conv')))
		h2 = lrelu( self.bn1( conv2d(h1, name='d_h2_conv')))
		h3 = lrelu( self.bn1( conv2d(h2, name='d_h3_conv')))
		h4 = linear( tf.reshape(h3, [-1, 8192]) , name='d_h3_lin')

		return tf.nn.sigmoid(h4), h4

	def generator(self, z):

		h0 = 
		h1 = 
		h2 = 
		h3 = 
		h4 = 

		return tf.nn.tanh(h4)



	def model(self):

		with tf.name_scope('input'):

			self.z = tf.placeholder(tf.float32)
			self.image = tf.placeholder(tf.float32)

		with tf.variable_scope('generator'):
			
			self.G = self.generator(self.z)

		with tf.variable_scope('discriminator'):
			
			self.D, self.D_logits = self.discriminator(self.image)
			self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
			
		with tf.name_scope('loss'):

			self.d_loss_real = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
			self.d_loss_fake = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
			self.d_loss = self.d_loss_fake + self.d_loss_real
			self.g_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

		with tf.name_scope('opt')

			t_vars = tf.trainable_variables()
			self.d_vars = [var for var in t_vars if 'd_' in var.name]
			self.g_vars = [var for var in t_vars if 'g_' in var.name]

			self.d_opt = tf.train.AdamOptimizer(0.01).minimize(self.d_loss, self.d_vars)
			self.g_opt = tf.train.AdamOptimizer(0.01).minimize(self.g_loss, self.d_vars)

		with tf.name_scope('summary'):

			self.g_loss_summary = tf.scalar_summary('d_loss', self.d_loss)
			self.d_loss_summary = tf.scalar_summary('g_loss', self.g_loss)

			self.G_summary = tf.image_summary('G', self.G)
			self.d_summary = tf.histogram_summary('d', self.D)
			self.d__summary = tf.histogram_summary('d_', self.D_)

def main():

	model = DCGAN()

	data_load = data_loader()

	with tf.Session() as sess:

		sess.run( tf.initialize_all_variables() )
		writer = tf.train.SummaryWriter('./logs', sess.graph)

		for step in range(100):

			z, image = data_load.load_next_batch()

			feed_dict = { model.z: z, model.image: image}

			sess.run([model.d_opt, model.d_opt], feed_dict = feed_dict)

			sess.run([model.g_opt], feed_dict = feed_dict)


			writer.add_summary(summary, step)

if __name__ == '__main__':
	main()
