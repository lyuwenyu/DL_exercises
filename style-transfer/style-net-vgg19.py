import tensorflow as tf 
import numpy as np 
import scipy.io 
from skimage import io, transform

def vgg19(input_data, model_path='imagenet-vgg-verydeep-19.mat'):
	layers = (
		'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
		
		'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

		'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
		'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

		'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
		'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'				
		)
	net = {}
	data = scipy.io.loadmat(model_path)
	mean_pixel = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
	weights = data['layers'][0]
	
	current = input_data
	#net['input'] = tf.Variable(tf.zeros([1,224,224,3]))
	for i, name in enumerate(layers):
		kind = name[:4]
		if kind == 'conv':
			assert str(weights[i][0][0][0][0]) == name
			kernel, bias = weights[i][0][0][2][0]
			kernel = np.transpose(kernel, [1, 0, 2, 3])
			bias = bias.reshape(-1)
			current = _conv_layer(current, kernel, bias, scopex=name)
		elif kind == 'relu':
			assert str(weights[i][0][0][0][0]) == name
			current = tf.nn.relu(current)
		elif kind == 'pool':
			assert str(weights[i][0][0][0][0]) == name
			current = _pool_layer(current)
		net[name] = current
	assert len(net) == len(layers)
	return net

def _conv_layer(inputs, weights, bias, scopex):
	conv = tf.nn.conv2d(inputs, weights, strides=(1,1,1,1), padding='SAME')
	return tf.nn.bias_add(conv, bias)

def _pool_layer(inputs):
	return tf.nn.max_pool(inputs, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')

def preprocess(image, mean_pixel):
	return image - mean_pixel

def deprocess(image, mean_pixel):
	return image + mean_pixel



mean_pixel = np.array([123.68, 116.779, 103.939])#.reshape((1,1,1,3))
content = io.imread('./1-content.jpg')
style = io.imread('./1-style.jpg')
content = np.expand_dims(content, 0) - mean_pixel
style = np.expand_dims(style, 0) - mean_pixel
style_shape = style.shape
content_shape = content.shape

#content_image = tf.placeholder(dtype=tf.float32, shape=[1, 224,224,3])
#style_image = tf.placeholder(dtype=tf.float32, shape=[1, 224,224,3])
#content_image = np.random.randn(1, 224,224,3)
#style_image = np.random.randn(1, 224,224,3)

#image = tf.Variable(tf.random_normal([1, 224, 224, 3]))

content_layer = 'relu5_1'
style_layers = ['relu1_1', 'relu2_1', 'relu3_1',  'relu5_1']

content_features = {}
style_features = {}


#graph1 = tf.Graph()
with tf.Graph().as_default() as graph1:

	data = tf.placeholder(tf.float32)
	net = vgg19(input_data=data)
	content_features[content_layer] = net[content_layer]
	for layer in style_layers:
		style_features[layer] = net[layer]

with tf.Session(graph = graph1) as sess:
	sess.run( tf.initialize_all_variables())
	content_features = sess.run(content_features, feed_dict= {data: content})
	style_features = sess.run(style_features, feed_dict= {data: style})

	for layer in style_features.keys():
		fea = style_features[layer]
		fea = np.reshape(fea, (-1, fea.shape[3]))
		gram = np.matmul(fea.T, fea) 
		style_features[layer] = gram


graph2 = tf.Graph()
with graph2.as_default(): #with g.control_dependencies([content_features, style_features])

	#image = tf.Variable(tf.random_normal(content_shape, mean=0.0)*0.256)
	image = tf.Variable(tf.random_uniform(content_shape,-20.0, 20.0))
	net =  vgg19(image)

	#content loss
	c_shape = content_features[content_layer].shape
	M = c_shape[1] * c_shape[2]
	N = c_shape[3]
	kx = 1.0 / (2* M**0.5 * N**0.5)
	content_loss = kx * tf.nn.l2_loss(net[content_layer]-content_features[content_layer])
	#tf.add_to_collection('loss', content_loss)

	#style loss
	style_losses = []
	for i, layer in enumerate(style_layers):
		layer_style = net[layer]
		_, height, width, channel = layer_style.get_shape().as_list()
		size = height * width
		k = 1.0 / (4.0 * size**2 * channel**2)
		feats = tf.reshape(layer_style, (-1, channel))
		gram = tf.matmul(tf.transpose(feats), feats)
		style_losses.append(k*tf.nn.l2_loss(gram-style_features[layer]))
		#tf.add_to_collection('loss', style_losses[i])

	loss = 0.1*content_loss + 0.01*tf.reduce_mean(style_losses)
	#loss = tf.reduce_sum(tf.get_collection('loss'))
	tf.scalar_summary('loss', loss)
	summary = tf.merge_all_summaries()

	train_step = tf.train.AdamOptimizer(5.0).minimize(loss)

	saver = tf.train.Saver()

with tf.Session(graph=graph2) as sess:
	sess.run(tf.initialize_all_variables())
	#saver.restore(sess, './models/style_100_all.ckpn')
	summary_writer = tf.train.SummaryWriter('./logs', graph=sess.graph)

	for i in xrange(1000):
		_, lossx, c,s, sumarr = sess.run([train_step, loss, content_loss, style_losses, summary])
		summary_writer.add_summary(sumarr, i)

		print lossx, i
		print c,s

		if i%10 == 0:
			saver.save(sess, 'models/style_{}.ckpn'.format(i))

			img = sess.run(image)
			img = np.squeeze(img)
			img = img + mean_pixel
			img = np.clip(img, 0, 255).astype(np.uint8)
			io.imsave('./result_{}.jpg'.format(i), img)
