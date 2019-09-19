import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
plt.style.use('ggplot')


class LeastSquares:
    
    def __init__(self, N = 100, method = 'ols', noise_magnitude = 0.01, polynomial_degree = 5):
        self.number_of_points = N
        self.method = method

        self.training_data = None
        self.test_data = None
        
        self.x = None
        self.y = None
        
        ##Sets self.X and self.y using 
        self.init_data(noise_magnitude)
        
        self.design_matrix = self.create_design_matrix(self.x, polynomial_degree)
        
        self.find_fit(method, split_data=True)
            
    def find_fit(self, method, split_data = False):
        """
        Use the selected method to find the best fit for the data.
        
        Returns:
            nothing, sets self.z_tilde
        """
        ## TODO: split data into training and test-data if specified.
        if split_data:
            y = np.ravel(self.y)
            design_train, design_test, y_train, y_test = train_test_split(self.design_matrix, y, test_size=0.2)
            y_tilde = self.ordinary_least_squares(design_train, y_train)
            y_predict = self.ordinary_least_squares(design_test, y_test, training_matrix=design_train)
            
            self.test_error_analysis(y_train, y_tilde)
            self.test_error_analysis(y_test, y_predict)
            
        else:
            y = self.y
            x = self.x

            
            
            if method == 'ols':
                self.y_tilde = self.ordinary_least_squares(self.training_data, self.design_matrix)
            else:
                print('invalid method.')
                return
        
            
    def init_data(self, noise):
        """
        Initialize dataset using franke's function with given noise
        magnitude and number of points on each axis.
        Returns:
            x, y, z -- Franke's function z(x, y) with added noise
        """
        ## Create uniform grid
        x = np.sort(np.random.rand(self.number_of_points))
        y = np.sort(np.random.rand(self.number_of_points))

        x, y = np.meshgrid(x, y)
        
        self.x = np.array([x, y])
        
        ## Compute franke's function with noise
        z = self.frankes_function(x, y, noise)
        
        self.y = z
        
        
    
    def ordinary_least_squares(self, design_matrix, z, training_matrix = [], training_y = []):
        """
        Performs ordinary least squares on given data
        
        Returns:
            z_tilde -- an array of same shape as z, containing values 
                       corresponding to the fitted function.
        """
        z1 = z
        if len(z.shape) > 1:
            z1 = np.ravel(z)
        
        if len(training_matrix) == 0 or len(training_y) == 0:
            beta = np.linalg.lstsq(design_matrix, z1, rcond=None)[0]
        else:
            beta = np.linalg.lstsq(training_matrix, training_y, rcond=None)[0]
            
        z_tilde = np.dot(beta, design_matrix.T)
        
        return z_tilde
        
        
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

    def create_design_matrix(self, X, n=5):
        """
        Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
        Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
        """
        x, y = X
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        l = int((n+1)*(n+2)/2)		# Number of elements in beta
        X = np.ones((N, l))

        for i in range(1, n+1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                X[:, q+k] = x**(i-k) * y**k

        return X

    def plot(self):
        """
        Displays two 3-dimensional plots, one of the original data and
        one of the computed best fit.
        """
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        surf = ax.plot_surface(self.X[0], self.X[1], self.y, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)


        ax = fig.add_subplot(1, 2, 2, projection='3d')
        surf = ax.plot_surface(self.X[0], self.X[1], self.y_tilde, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        
    def mean_squared_error(self, z, z_tilde):
        if len(z.shape) > 1:
            z = np.ravel(z)
            z_tilde = np.ravel(z_tilde)
            
        return np.mean((z - z_tilde)**2)
    
    def r2_score(self, z, z_tilde):
        if len(z.shape) > 1:
            z = np.ravel(z)
            z_tilde = np.ravel(z_tilde)
            
        return 1 - np.sum((z - z_tilde) ** 2) / np.sum((z - np.mean(z)) ** 2)
    
    def test_error_analysis(self, y, y_tilde):
        """
        Compares the manual calculations of Mean Squared Error
        and R-squared score with the ones calculated using sklearn.
        """
        y = y.ravel()
        y_tilde = y_tilde.ravel()
        print("-")
        print(f"MSE(manual): {self.mean_squared_error(y, y_tilde)}")
        print(f"MSE(sklearn): {metrics.mean_squared_error(y, y_tilde)}")
        print("-")
        print(f"R^2 Score(manual): {self.r2_score(y, y_tilde)}")
        print(f"R^2 Score(sklearn): {metrics.r2_score(y, y_tilde)}")
        print("-")
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='method', type=str, help='Regression method', default='ols')
    parser.add_argument('-degree', dest='poly_degree', type=int, help='Polynomial degree for design matrix', default=5)
    parser.add_argument('-noise', dest='noise_magnitude', type=float, help='Magnitude for the noise added to Frankes function', default=0.01)
    parser.add_argument('-n', dest='N', type=int, help='Number of points in x- and y-directions', default=100)
    parser.add_argument('--test', help='Runs test functions', action='store_true')
    parser.add_argument('--plot', help='Plots the resulting functions side by side', action='store_true')
    args = parser.parse_args()
    
    fit = LeastSquares(args.N, args.method, args.noise_magnitude, args.poly_degree)
    
    if args.plot:
        fit.plot()
    if args.test:
        fit.test_error_analysis()
    
    
