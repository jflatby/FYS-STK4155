
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.style.use('ggplot')


def frankes_function(x, y, noise_magnitude=0.01):
    """
    Franke's test function with added noise
    :param x: array of x-values
    :param y: array of y-values
    :return: z-value corresponding to given x and y
    """
    return 0.75*np.exp(-(9*x - 2)**2 / 4 - (9*y - 2)**2 / 4) \
           + 0.75*np.exp(-(9*x + 1)**2 / 49 - (9*y + 1) / 10) \
           + 0.5*np.exp(-(9*x - 7)**2 / 4 - (9*y - 3)**2 / 4) \
           - 0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2) \
           + noise_magnitude * np.random.randn(len(x))



def create_design_matrix(x, y, n = 5):
	"""
	Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
	Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
	"""
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = x**(i-k) + y**k

	return X



number_of_points = 100
x = np.sort(np.random.rand(number_of_points))
y = np.sort(np.random.rand(number_of_points))

x, y = np.meshgrid(x, y)

z = frankes_function(x, y)

x_1, y_1 = np.ravel(x), np.ravel(y)
n = int(len(x_1))
z_1 = np.ravel(z) + np.random.random(n) * 1
design_matrix = create_design_matrix(x_1, y_1)


fit = np.linalg.lstsq(design_matrix, z_1, rcond = None)[0]
z_tilde = np.dot(fit, design_matrix.T)
print(np.shape(z))


fig = plt.figure(figsize = plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)


ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(x, y, np.reshape(z_tilde, (number_of_points, number_of_points)), cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)


#plt.show()
