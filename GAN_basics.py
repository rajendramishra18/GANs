import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def xavier_init(size):
	in_dim = size[0]
	xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
	return tf.random_normal(shape=size, stddev=xavier_stddev)


'''
Tensorflow:
placeholder : A placeholder is simply a variable that we will assign data to at a later date.
 It allows us to create our operations and build our computation graph, without needing the data.
 
'''

'''
Let us take a variable X which works as input in this case.
None spacifies that X can have any number of rows, but 784 columns.
each value in X will be of type tf.float
'''
X = tf.placeholder(tf.float32, shape=[None, 784])

'''
In GANs we have a Generator and a Discriminator Network.
These networks are basically neural net based architectures.
For simplicity let us assume that we have a simple model.
'''

''' 
For Discriminator we have two layers.
First layer takes X as input. 
We will have 784 neurons as input and 128 neurons as receptors.
So total 784*128 learning parameters
D_W1 * X + D_b1
Discriminator's ultimate work is to discriminate whether X is coming form original distribution or from forged distribution.
'''

D_W1 = tf.Variable(xavier_init([784, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(xavier_init([128,1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1,D_W2,D_b1,D_b2]

'''
Let Z be the Morphed replica of X.
Generator will try to generate the morphed copies of X which as similar to X.
'''

Z = tf.placeholder(tf.float32,shape=[None,100])

G_W1 = tf.Variable(xavier_init([100, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))

G_W2 = tf.Variable(xavier_init([128, 784]))
G_b2 = tf.Variable(tf.zeros(shape = [784]))

theta_G = [G_W1,G_W2, G_b1, G_b2]

'''
Define the loss function for Discriminator and Generator.
Discriminator tries to maximise the probablity of Real Image and tries to reduce the probability of Fake image.
Contrary to this, Generator tries to maximise the probability of Fake Image getting selected as real image.
For each real image, we have already computed D_real.
For each fake image sample, we have D_fake.
''' 
def define_loss_function_1(D_real, D_fake):
	D_loss = -tf.reduce_mean(tf.log(D_real),tf.log(1.0 - D_fake))
	G_loss = -tf.reduce_mean(tf.log(D_fake))
	return D_loss, G_loss
	
'''
Another way this function could be thought like:
Cross_entropy loss fucntion is given by p*log(1-p) or p*logq 
Here p is the expected output and q is the actual output.
let us say, our labels are [0,1]. 1 is used to denote real image and 0 is used to denote fake image. 

For Discriminator:
So in case of real images, we would like the real logits to be as close as possible to 1.
In similar way, we would like the fake logits to be as close as possible to 0.

For Generator:
Generator would like the fake logits to be as close as to 1 so that Discriminator will accept it.
'''
def define_loss_function_2(D_real_logits, D_fake_logits):
	D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real_logits , labels = tf.ones_like(D_real_logits)))
	D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_logits , labels = tf.zeros_like(D_fake_logits)))
	D_loss = D_loss_real+D_loss_fake
	G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_logits , labels = tf.ones_like(D_fake_logits)))
	return D_loss, G_loss
	
	
'''
Now let us see what will generator do given X.
'''
def generator(z):
	'''
	z is sampled from a random distribution. We will get values of hidden neurons G_h1.
	'''
	G_h1 = tf.nn.relu(tf.matmul(z,G_W1)+G_b1)
	
	'''
	For the second layer, output of hidden neurons is the input and after multiplying it by G_W2 we will get log probability.
	'''
	G_log_prob = tf.matmul(G_h1,G_W2) + G_b2
	
	'''
	And then finally, to scale the values in 0-1 range, apply sigmoid.
	'''
	G_prob = tf.nn.sigmoid(G_log_prob)
	
	return G_prob
	
	
def discriminator(x):
	'''
	x comes from the original distribution pdata.
	'''
	D_h1 = tf.nn.relu(tf.matmul(x,D_W1)+D_b1)
	
	D_logit = tf.matmul(D_h1,D_W2)+D_b2
	
	D_prob = tf.nn.sigmoid(D_logit)
	
	return D_prob,D_logit
	
	
def sample_Z(m, n):
	return np.random.uniform(-1., 1., size=[m, n])
	
def plot(samples):
	fig = plt.figure(figsize=(4, 4))
	gs = gridspec.GridSpec(4, 4)
	gs.update(wspace=0.05, hspace=0.05)
	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
	return fig
	
''' Generate Z - a fake image using generator '''

G_sample = generator(Z)

''' Get real and fake probabilities '''
D_real , D_real_logit = discriminator(X)

D_fake , D_fake_logit = discriminator(G_sample)

''' Get the objective functions for learning for Generator and Discriminator '''
#D_loss,G_loss = define_loss_function_1(D_real,D_fake)

D_loss,G_loss = define_loss_function_2(D_real_logit,D_fake_logit)


''' Apply optimizer on these loss functions'''

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list = theta_G)

''' Load MNIST dataset'''
mnist = input_data.read_data_sets("/home/raj/projects/tensorflow/GANs/data", one_hot=True)

mb_size = 128
Z_dim = 100

sess = tf.Session()
sess.run(tf.global_variables_initializer())

''' Check if the directory for output present. If not create one '''
if not os.path.exists("out/"):
	os.makedirs("out/")

i = 0
for it in range(1000000):
	X_mb, _ = mnist.train.next_batch(mb_size)
	_, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
	_, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})
	if it % 1000 == 0:
		samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
		fig = plot(samples)
		plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
		i += 1
		plt.close(fig)
		print('Iter: {}'.format(it))
		print('D loss: {:.4}'. format(D_loss_curr))
		print('G_loss: {:.4}'.format(G_loss_curr))
		print()

