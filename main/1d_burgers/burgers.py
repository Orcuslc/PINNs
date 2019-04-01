import tensorflow as tf
import numpy as np
import time

class PINN:
	# Burgers Equation u = u(x, t)

	def __init__(self, x_initial, u_initial, t_left, t_right, u_left, u_right, collocation_point, layers, log_file):

		# x points on t = 0.
		self.x_initial = x_initial
		self.t_initial = x_initial*0
		self.u_initial = u_initial

		# t points on left boundary: x = -1 and right boundary: x = 1
		self.t_left = t_left
		self.t_right = t_right
		self.x_left = -1.*np.ones_like(self.t_left)
		self.x_right = np.ones_like(self.t_right)
		self.u_left = u_left
		self.u_right = u_right

		# collocation points of x and t (to use in the residue net)
		self.x_collocation = collocation_point[:, :1]
		self.t_collocation = collocation_point[:, 1:]

		self.layers = layers
		self.log_file = log_file

		# input
		self.min = -1.
		self.max = 1.
		self._construct()

	def _initialize_NN(self, layers):
		weights = []
		biases = []
		for i in range(len(layers)-1):
			W = self._xavier_init(size = [layers[i], layers[i+1]])
			b = tf.Variable(tf.zeros([1, layers[i+1]], dtype = tf.float32), dtype = tf.float32)
			weights.append(W)
			biases.append(b)
		return weights, biases

	def _xavier_init(self, size):
		[in_dim, out_dim] = size
		xavier_stddev = np.sqrt(2/(in_dim + out_dim))
		return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev = xavier_stddev), dtype = tf.float32)

	def neural_network(self, X, weights, biases):
		# normalize
		X = 2.0*(X - self.min)/(self.max - self.min) - 1.0

		# forward pass
		for i in range(len(weights)-2):
			X = tf.tanh(tf.add(tf.matmul(X, weights[i]), biases[i]))
		Y = tf.add(tf.matmul(X, weights[-1]), biases[-1])
		return Y

	def net_u(self, t, x):
		return self.neural_network(tf.concat([t, x], 1), self.weights, self.biases) 

	def net_residue(self, t, x):
		u = self.neural_network(tf.concat([t, x], 1), self.weights, self.biases)
		u_t = tf.gradients(u, t)[0]
		u_x = tf.gradients(u, x)[0]
		u_xx = tf.gradients(u_x, x)[0]
		f = u_t + u*u_x - (0.01/np.pi)*u_xx
		return f

	def _construct(self):
		# placeholders
		self._x_initial = tf.placeholder(tf.float32, shape = [None, self.x_initial.shape[1]], name = 'x_initial')
		self._t_initial = tf.placeholder(tf.float32, shape = [None, self.t_initial.shape[1]], name = 't_initial')
		self._u_initial = tf.placeholder(tf.float32, shape = [None, self.u_initial.shape[1]], name = 'u_initial')

		self._x_left = tf.placeholder(tf.float32, shape = [None, self.x_left.shape[1]], name = 'x_left')
		self._x_right = tf.placeholder(tf.float32, shape = [None, self.x_right.shape[1]], name = 'x_right')
		self._t_left = tf.placeholder(tf.float32, shape = [None, self.t_left.shape[1]], name = 't_left')
		self._t_right = tf.placeholder(tf.float32, shape = [None, self.t_right.shape[1]], name = 't_right')
		self._u_left = tf.placeholder(tf.float32, shape = [None, self.u_left.shape[1]], name = 'u_left')
		self._u_right = tf.placeholder(tf.float32, shape = [None, self.u_right.shape[1]], name = 'u_right')

		self._x_collocation = tf.placeholder(tf.float32, shape = [None, self.x_collocation.shape[1]], name = 'x_collocation')
		self._t_collocation = tf.placeholder(tf.float32, shape = [None, self.t_collocation.shape[1]], name = 't_collocation')

		# initialize NN
		self.weights, self.biases = self._initialize_NN(self.layers)

		# computational graph
		self.u_initial_pred = self.net_u(self._t_initial, self._x_initial)
		self.u_left_pred = self.net_u(self._t_left, self._x_left)
		self.u_right_pred = self.net_u(self._t_right, self._x_right)
		self.residue_pred = self.net_residue(self._t_collocation, self._x_collocation)

		# loss
		self.loss = tf.reduce_mean(tf.square(self._u_initial - self.u_initial_pred)) + \
			tf.reduce_mean(tf.square(self._u_left - self.u_left_pred)) + \
			tf.reduce_mean(tf.square(self._u_right - self.u_right_pred)) + \
			tf.reduce_mean(tf.square(self.residue_pred))

		# optimizer
		self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
				method = 'L-BFGS-B',
				options = {
					'maxiter': 50000,
					'maxfun': 50000,
					'maxcor': 50,
					'maxls': 50,
					'ftol': 1.0+np.finfo(float).eps
					})
		self.optimizer_Adam = tf.train.AdamOptimizer()
		self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

		# session
		self.session = tf.Session(config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

		# save model
		self.session.run(tf.global_variables_initializer())


	def train(self, iterations):
		# key-values
		values = {self._x_initial: self.x_initial, self._x_left: self.x_left, self._x_right: self.x_right, self._t_initial: self.t_initial, self._t_left: self.t_left, self._t_right: self.t_right, self._u_initial: self.u_initial, self._u_left: self.u_left, self._u_right: self.u_right, self._x_collocation: self.x_collocation, self._t_collocation: self.t_collocation} 

		t_start = time.time()
		for it in range(iterations):
			self.session.run(self.train_op_Adam, values)

			if it % 100 == 0:
				loss = self.session.run(self.loss, values)
				print("Iteration: {0}, Loss: {1}, Time: {2}".format(it, loss, time.time() - t_start))
				with open(self.log_file, 'a+') as f:
					f.write("Iteration: {0}, Loss: {1}, Time: {2}\n".format(it, loss, time.time() - t_start))

		self.optimizer.minimize(self.session, feed_dict = values, fetches = [self.loss])
		# tf.saved_model.simple_save(self.session, 'model.tf', inputs = {
		# 	'x': self._x,
		# 	't': self._t
		# 	},
		# 	outputs = {
		# 	'u_pred': self.u_pred,
		# 	'f_pred': self.f_pred
		# 	})

	def predict(self, x, t):
		# predict u = u(t, x)
		values_u = {self._x_initial: x, self._t_initial: t}
		u = self.session.run(self.u_initial_pred, values_u)

		values_residue = {self._x_collocation: x, self._t_collocation: t}
		residue = self.session.run(self.residue_pred, values_residue)

		return u, residue

if __name__ == '__main__':
	layers = [2] + [20]*8 + [1]

	# training data: 25 on each boundary, 50 on initial
	np.random.seed(1)

	# boundary
	N_left = 25
	N_right = 25
	N_initial = 50
	t_left = np.expand_dims(np.random.random(N_left), 1)
	t_right = np.expand_dims(np.random.random(N_right), 1)
	x_left = np.expand_dims(np.ones(N_left)*-1, 1)
	x_right = np.expand_dims(np.ones(N_right)*1, 1)
	u_left = np.expand_dims(np.zeros(N_left), 1)
	u_right = np.expand_dims(np.zeros(N_right), 1)

	# initial
	x_initial = np.expand_dims(np.random.random(N_initial)*2-1, 1)
	t_initial = np.expand_dims(np.zeros(N_initial), 1)
	u_initial = -np.sin(np.pi * x_initial)

	# collocation points
	from pyDOE import lhs
	N_collocation = 10000
	collocation_point = lhs(2, N_collocation)
	collocation_point[:, 0] = collocation_point[:, 0]*2-1 # x in [-1, 1] 


	# train
	model = PINN(x_initial, u_initial, t_left, t_right, u_left, u_right, collocation_point, layers, 'log.txt')
	model.train(1000)

	## test

	# load test data
	from scipy.io import loadmat

	# reshape by columns, i.e., [[x1, t1], [x2, t1], ..., [x1, t2], ..]
	u_test = loadmat("burgers.mat")["u"].reshape((-1, 1), order = 'F')

	x_test_1d = np.linspace(-1., 1., 201)
	t_test_1d = np.linspace(0., 1., 2001)

	x_test = np.expand_dims(np.tile(x_test_1d, 2001), 1)
	t_test = np.expand_dims(np.asarray([[i]*201 for i in t_test_1d]).flatten(), 1)

	u_pred, residue = model.predict(x_test, t_test)
	np.save("u_pred", u_pred)
	print(residue)

	MSE = np.mean((u_pred - u_test)**2)
	RMSE = np.mean(((u_pred - u_test)/u_test)**2)
	print(MSE, RMSE)

	# import pickle
	# f = open('model', 'wb')
	# pickle.dump(model, f)
