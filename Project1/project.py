import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sklearn.metrics as metrics
plt.style.use('ggplot')


class Project1:
    
    def __init__(self, N = 100, method = 'ols', noise_magnitude = 0.01, polynomial_degree = 5):
        self.number_of_points = N
        self.method = method
        
        ## Create uniform grid
        x = np.sort(np.random.rand(N))
        y = np.sort(np.random.rand(N))

        self.x, self.y = np.meshgrid(x, y)
        
        ## Compute franke's function with noise
        self.z = self.frankes_function(self.x, self.y, noise_magnitude)
        
        ## create design_matrix
        self.design_matrix = self.create_design_matrix(self.x, self.y, polynomial_degree)
        
        if method == 'ols':
            self.z_tilde = self.ordinary_least_squares(self.z, self.design_matrix)
        else:
            print('invalid method.')
            return
            
    def ordinary_least_squares(self, z, design_matrix):
        n = self.number_of_points * self.number_of_points
        z_1 = np.ravel(z)

        fit = np.linalg.lstsq(design_matrix, z_1, rcond=None)[0]
        z_tilde = np.dot(fit, design_matrix.T)
        
        return np.reshape(z_tilde, (self.number_of_points, self.number_of_points))
        
        
    def frankes_function(self, x, y, noise_magnitude=0.01):
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

    def create_design_matrix(self, x, y, n=5):
        """
        Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
        Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
        """
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        l = int((n+1)*(n+2)/2)		# Number of elements in beta
        X = np.ones((N, l))

        for i in range(1, n+1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                X[:, q+k] = x**(i-k) + y**k

        return X

    def plot(self):
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        surf = ax.plot_surface(self.x, self.y, self.z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)


        ax = fig.add_subplot(1, 2, 2, projection='3d')
        surf = ax.plot_surface(self.x, self.y, self.z_tilde, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

        #fig = plt.figure()
        #ax = fig.gca(projection='3d')
        #surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        #fig.colorbar(surf, shrink=0.5, aspect=5)


        # plt.show()
        
    def mean_squared_error(self, z, z_tilde):
        return np.mean((z - z_tilde)**2)
    
    def r2_score(self, z, z_tilde):
        return 1 - np.sum((z - z_tilde) ** 2) / np.sum((z - np.mean(z_tilde)) ** 2)
    
    def test_error_analysis(self):
        print(f"MSE(manual): {self.mean_squared_error(self.z, self.z_tilde)}")
        print(f"MSE(sklearn): {metrics.mean_squared_error(self.z, self.z_tilde)}")
        print("-")
        print(f"R^2 Score(manual): {self.r2_score(self.z, self.z_tilde)}")
        print(f"R^2 Score(sklearn): {metrics.r2_score(self.z, self.z_tilde)}")
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='method', type=str, help='Regression method', default='ols')
    parser.add_argument('-degree', dest='poly_degree', type=int, help='Polynomial degree for design matrix', default=5)
    parser.add_argument('-noise', dest='noise_magnitude', type=float, help='Magnitude for the noise added to Frankes function', default=0.01)
    parser.add_argument('-n', dest='N', type=int, help='Number of points in x- and y-directions', default=100)
    parser.add_argument('--test', help='Runs test functions', action='store_true')
    parser.add_argument('--plot', help='Plots the resulting functions side by side', action='store_true')
    args = parser.parse_args()
    
    project = Project1(args.N, args.method, args.noise_magnitude, args.poly_degree)
    
    if args.plot:
        project.plot()
    if args.test:
        project.test_error_analysis()
    
    
