import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numexpr as ne
import random
import gradient_descent

def initialize_params():
	return [-8, -8] #[m, b]

def calculate_error(params):
	m = params[0]
	b = params[1]
	error = 0
	for point in data_points:
		y = point[0]
		x = point[1]
		error = error + (y - (m*x + b))**2
	return error/float(len(data_points))

def calculate_derivs(data_points, params):
	m = params[0]
	b = params[1]
	derivs = [0, 0]

	for point in data_points:
		y = point[0]
		x = point[1]
		derivs[0] = derivs[0] + x*(y - (m*x + b))
		derivs[1] = derivs[1] + (y - (m*x + b))

	factor = -2.0/len(data_points)
	for i in range(0, len(derivs)):
		derivs[i] = factor * derivs[i]
	
	return derivs

def plot_error_surface(ax):
	m = b = np.arange(-8.0, 20.0, 0.05)
	M, B = np.meshgrid(m, b)
	error = np.array([calculate_error([m, b]) for m, b in zip(np.ravel(M), np.ravel(B))])
	Error = error.reshape(M.shape)
	ax.plot_surface(M, B, Error, cmap = 'gist_rainbow_r', alpha = 0.25)


def save_line_graph(m, b, data_points, out_file):
	fig = plt.figure()
	subplot = fig.add_subplot(111)
	subplot.set_xlabel('x')
	subplot.set_ylabel('y')
	subplot.set_title('m = %.2f    b = %.2f' % (m, b))
	for point in data_points:
		y = point[0]
		x = point[1]
		subplot.plot(x, y, 'bo', clip_on = False)

	x = np.linspace(-2.0, 2.0, 100)
	y = ne.evaluate('m*x + b', local_dict = {'x': x, 'm': m, 'b': b})

	subplot.plot(x, y, 'r')
	subplot.grid()
	plt.ylim([-15, 15])
	plt.xlim([-2, 2])

	plt.savefig(out_file, bbox_inches='tight', pad_inches=0.0)


data_points = np.load('example_1_sample_points.npy')

parameters = gradient_descent.run(data_points, initialize_params, calculate_derivs, calculate_error)

print('plotting error surface and steps...(this could take a few minutes)')
count = 0
for i in range(0, len(parameters)):
	error_fig = plt.figure()
	error_surface = error_fig.add_subplot(111, projection = '3d')
	#Plot original error surface
	plot_error_surface(error_surface)
	error_surface.set_xlabel('m')
	error_surface.set_ylabel('b')
	error_surface.set_zlabel('Error')
	error_surface.view_init(elev = 40, azim = 79)	

	err = 0
	for j in range(0, i + 1):
		param = parameters[j]
		m = param[0][0]
		b = param[0][1]
		err = param[1]
		#draw point
		error_surface.scatter(m, b, err ,color = "r", s = 25)	
	
	error_surface.set_title('Error = %.2f' % err)
	plt.savefig('error_surface-' + str(count) + '.png', bbox_inches='tight', pad_inches=0.0)
	count = count + 1

print('plotting line graphs')
count = 0
for params in parameters:
	m = params[0][0]
	b = params[0][1]
	save_line_graph(m, b, data_points, 'line_graph-' + str(count) + '.png')
	count = count + 1

np.save('parameter_estimates_example_1.npy', parameters)