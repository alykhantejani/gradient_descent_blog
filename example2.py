import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numexpr as ne
import random
import gradient_descent

def initialize_params():
	#return [-2.5, -2.5, -2.5, -2.5, -2.5, -2.5] #[a, b, c, d, e, f]
	#return [0, 1, 0, -1/6.0, 0, 1/120.0] #[a, b, c, d, e, f]
	#return [0, 1, 0, 0, 0, 0] #[a, b, c, d, e, f]
	return [0,0,0, 0]

def calculate_error(params):
	a, b, c, d = params[0], params[1], params[2], params[3]
	
	error = 0
	for point in data_points:
		y = point[0]
		x = point[1]
		error = error + (y - (a + b*x + c*x**2 + d*x**3))**2
	return error/float(len(data_points))

def calculate_derivs(data_points, params):
	a, b, c, d = params[0], params[1], params[2], params[3]
	derivs = [0, 0, 0, 0]

	for point in data_points:
		y = point[0]
		x = point[1]
		inner = y - (a + b*x + c*x**2 + d*x**3)
		derivs[0] = derivs[0] + inner
		derivs[1] = derivs[1] + x*inner
		derivs[2] = derivs[2] + (x**2)*inner
		derivs[3] = derivs[3] + (x**3)*inner

	factor = -2.0/len(data_points)
	for i in range(0, len(derivs)):
		derivs[i] = factor * derivs[i]
	
	return derivs

def save_line_graph(param, err, data_points, out_file):
	a, b, c, d = param[0], param[1], param[2], param[3]

	fig = plt.figure()
	subplot = fig.add_subplot(111)
	subplot.set_xlabel('x')
	subplot.set_ylabel('y')
	subplot.grid()
	subplot.set_title('a = %.2f, b = %.2f, c = %.2f, d = %.2f Error = %.2f' % (a, b, c, d, err))

	min_y = float('inf')
	max_y = float('-inf')

	for point in data_points:
		y = point[0]
		x = point[1]
		subplot.plot(x, y, 'bo', clip_on = False)
		min_y = min(y, min_y)
		max_y = max(y, max_y)

	x = np.linspace(-2.5, 2.5, 100)
	y = ne.evaluate('a + b*x + c*x**2 + d*x**3', local_dict = {'x': x, 'a': a, 'b': b, 'c' : c, 'd' : d})

	subplot.plot(x, y, 'r')
	plt.ylim([min_y, max_y])
	plt.xlim([-2.5, 2.5])
	plt.savefig(out_file, bbox_inches='tight', pad_inches=0.0)


data_points = np.load('example_2_sample_points.npy')

parameters = gradient_descent.run(data_points, initialize_params, calculate_derivs, calculate_error, learning_rate = 0.03, max_steps = 100)

print('plotting line graphs')
count = 0
for params in parameters:
	param = params[0]
	err = params[1]
	save_line_graph(param, err, data_points, 'line_graph-' + str(count) + '.png')
	count = count + 1

np.save('parameter_estimates_example_2.npy', parameters)