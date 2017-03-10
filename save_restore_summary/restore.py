import tensorflow as tf
import numpy as np




saver = tf.train.import_meta_graph('./models/model-2000.meta')


graph = tf.get_default_graph()


w = graph.get_tensor_by_name('W:0')
b = graph.get_tensor_by_name('b:0')
names = [op.name for op in graph.get_operations()]

p = graph.get_tensor_by_name('input_data/Placeholder:0')
p1 = graph.get_tensor_by_name('input_data/Placeholder_1:0')
#p1 = graph.get_operation_by_name('input_data/Placeholder_1')

loss = graph.get_operation_by_name('loss')

#pre = graph.get_operation_by_name('model/add')
pre = tf.get_collection('pre')

with tf.Session(graph=graph) as sess:
	
	#print sess.run(ops)
	saver.restore(sess, './models/model-2000')
	
	print sess.run(w )
	print sess.run(b ) 
	#print sess.run(names )
	
	fd = {p: [[11.0]]}
	b = sess.run( pre, feed_dict= fd)
	print b

	for n in names:
		pass
		#print n
