
# reference https://github.com/NELSONZHAO/zhihu/tree/master/mnist_gan

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)

LOGDIR = "./log/mnist_gan" 

real_img_size = mnist.train.images[0].shape[0]

noise_img_size = 100

g_units = 128

d_units = 128

alpha = 0.1

learning_rate = 0.001

smooth = 0.05

# train
batch_size = 100
k = 10

epochs = 300


def get_inputs(real_img_size, noise_img_size):
	"""
	read image tensor and noise image tensor, as well as image digit
	"""
	real_img = tf.placeholder(tf.float32,
		shape = [None, real_img_size], name = "real_img")

	noise_img = tf.placeholder(tf.float32,
		shape = [None, noise_img_size], name = "noise_img")

	return real_img, noise_img

def get_sample(sample_shape):
	"""
	generator output noise image
	"""
	noise = np.random.normal(0.0, 1.0, sample_shape)
	return noise

def get_generator(noise_img, n_units, out_dim, reuse = False, alpha = 0.01):
	"""
	generator

	noise_img: input of generator
	n_units: # hidden units
	out_dim: # output
	alpha: parameter of leaky ReLU
	"""
	with tf.variable_scope("generator", reuse = reuse):
		# hidden layer
		hidden = tf.layers.dense(noise_img, n_units)
        # leaky ReLU
		relu = tf.maximum(alpha * hidden, hidden)
        # dropout
		drop = tf.layers.dropout(relu, rate = 0.5)

        # logits & outputs
		logits = tf.layers.dense(drop, out_dim)
		outputs = tf.tanh(logits)
        
		return logits, outputs


def get_discriminator(img, n_units, reuse = False, alpha = 0.01):
	"""
	discriminator

	n_units: # hidden units
	alpha: parameter of leaky Relu
	"""
	with tf.variable_scope("discriminator", reuse=reuse):
        # hidden layer
		hidden = tf.layers.dense(img, n_units)
		relu = tf.maximum(alpha * hidden, hidden)
        # dropout
		drop = tf.layers.dropout(relu, rate = 0.5)

        # logits & outputs
		logits = tf.layers.dense(drop, 1)
		outputs = tf.sigmoid(logits)
        
		return logits, outputs


with tf.Graph().as_default():

	real_img, noise_img = get_inputs(real_img_size, noise_img_size)

	# generator
	g_logits, g_outputs = get_generator(noise_img, g_units, real_img_size)

	sample_images = tf.reshape(g_outputs, [-1, 28, 28, 1])
	tf.summary.image("sample_images", sample_images, 10)

	# discriminator
	d_logits_real, d_outputs_real = get_discriminator(real_img, d_units)
	d_logits_fake, d_outputs_fake = get_discriminator(g_outputs, d_units, reuse = True)


	# discriminator loss
	d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_real, 
	                                                                     labels = tf.ones_like(d_logits_real)) * (1 - smooth))
	d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake, 
	                                                                     labels = tf.zeros_like(d_logits_fake)))
	# loss
	d_loss = tf.add(d_loss_real, d_loss_fake)

	# generator loss
	g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake,
	                                                                labels = tf.ones_like(d_logits_fake)) * (1 - smooth) )

	tf.summary.scalar("d_loss_real", d_loss_real)
	tf.summary.scalar("d_loss_fake", d_loss_fake)
	tf.summary.scalar("d_loss", d_loss)
	tf.summary.scalar("g_loss", g_loss)

	# optimizer
	train_vars = tf.trainable_variables()

	# generator tensor
	g_vars = [var for var in train_vars if var.name.startswith("generator")]
	# discriminator tensor
	d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

	# optimizer
	d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
	g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

	summary = tf.summary.merge_all()

	init = tf.global_variables_initializer()
	# save generator variables
	saver = tf.train.Saver()

	sess = tf.Session()

	summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)

	# Run the Op to initialize the variables.
	sess.run(init)

	for e in xrange(epochs):
		for i in xrange(mnist.train.num_examples//(batch_size * k)):
			for j in xrange(k):
				batch = mnist.train.next_batch(batch_size)

				# scale the input images
				images = batch[0].reshape((batch_size, 784))
				images = 2 * images - 1

				# generator input noises
				noises = get_sample([batch_size, noise_img_size])

				# Run optimizer
				sess.run([d_train_opt, g_train_opt],
					feed_dict = {real_img: images, noise_img: noises})
		
		# train loss
		images = 2 * mnist.train.images - 1.0
		noises = get_sample([mnist.train.num_examples, noise_img_size])

		summary_str, train_loss_d_real, train_loss_d_fake, train_loss_g = \
			sess.run([summary, d_loss_real, d_loss_fake, g_loss],
			feed_dict = {real_img: images, noise_img: noises})

		summary_writer.add_summary(summary_str, e)
		summary_writer.flush()
		
		train_loss_d = train_loss_d_real + train_loss_d_fake

		print("Epoch {}/{}".format(e+1, epochs),
			"Discriminator loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})".format(
				train_loss_d, train_loss_d_real, train_loss_d_fake),
			"Generator loss: {:.4f}".format(train_loss_g))

		# save checkpoints
		saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), global_step = e)

