import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

data_points = np.load('example_1_samples_points.npy')

def calc_error(m, b):
	error = 0
	for point in data_points:
		y = point[0]
		x = point[1]
		error = error + (y - (m*x + b))**2
	return error/len(data_points)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
m = b = np.arange(-8.0, 20.0, 0.05)
M, B = np.meshgrid(m, b)

error = np.array([calc_error(m, b) for m, b in zip(np.ravel(M), np.ravel(B))])

Error = error.reshape(M.shape)

ax.plot_surface(M, B, Error, cmap = 'gist_rainbow_r')

ax.set_xlabel('m')
ax.set_ylabel('b')
ax.set_zlabel('Error')

ax.view_init(elev = 39, azim = 24)
plt.show()