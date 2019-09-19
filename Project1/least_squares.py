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
        self.total_data_points = N * N
        self.method = method
        self.degree = polynomial_degree
        self.noise = noise_magnitude

        self.x = None
        self.y = None



    def init_franke(self):
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

        x = np.array([x, y])

        ## Compute franke's function with noise
        z = self.frankes_function(x, self.noise)

        return x, z



    def create_design_matrix(self, x):
        """
        Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
        Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
        """
        x, y = x
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        l = int((self.degree+1)*(self.degree+2)/2)		# Number of elements in beta
        X = np.ones((N, l))

        for i in range(1, self.degree+1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                X[:, q+k] = x**(i-k) * y**k

        return X



    def find_fit(self, data, design_matrix):
        """
        Use the selected method to find the best fit for the data.

        Returns:
            nothing, sets self.z_tilde
        """
        if len(data.shape) > 1:
            data = np.ravel(data)

        if self.method == 'ols':
            beta = self.ordinary_least_squares(design_matrix, data)
            y_tilde = np.dot(design_matrix, beta)
            return y_tilde
        else:
            print('invalid method.')
            return



    def cross_validation(self, x, data, k=5):
        """
        Implementation of the k-fold cross validation algorithm.
        """
        x = x.reshape(2, 10000)
        data = data.reshape(10000)

        fold_size = self.total_data_points // k

        mse_train, mse_test = np.zeros(k), np.zeros(k)




        for i in range(k):
            x_test, test_data = x[:, i*fold_size:(i+1)*fold_size], data[i*fold_size:(i+1)*fold_size]

            if i == 0:
                design_matrix_train = self.create_design_matrix(x[:, (i+1)*fold_size:])
                training_data = data[(i+1)*fold_size:]

            elif i == k-1:
                x_test, y_test = x[:, i*fold_size:], data[i*fold_size:]
                design_matrix_train = self.create_design_matrix(x[:, :i*fold_size])
                training_data = data[:i*fold_size]

            else:
                mask = np.array([True for i in range(x.shape[1])])
                mask[i*fold_size:(i+1)*fold_size] = False
                design_matrix_train = self.create_design_matrix(x[:, mask])
                training_data = data[mask]


            beta = self.ordinary_least_squares(design_matrix_train, training_data)
            training_prediction = np.dot(design_matrix_train, beta)

            design_matrix_test = self.create_design_matrix(x_test)
            test_prediction = np.dot(design_matrix_test, beta)

            mse_train[i] = self.mean_squared_error(training_data, training_prediction)
            mse_test[i] = self.mean_squared_error(test_data, test_prediction)


        mse_training_value = np.mean(mse_train)
        mse_test_value = np.mean(mse_test)


        print(mse_training_value, mse_test_value)



    def ordinary_least_squares(self, design_matrix, y):
        """
        Performs ordinary least squares on given data

        Returns:
            y_tilde -- an array of same shape as y, containing values
                       corresponding to the fitted function.
        """
        beta = np.linalg.lstsq(design_matrix, y, rcond=None)[0]

        #y_tilde = np.dot(beta, design_matrix.T)

        return beta



    def frankes_function(self, x, noise_magnitude=0.01):
        """
        Franke's test function with added noise
        :param x: array of x-values
        :param y: array of y-values
        :return: z-value corresponding to given x and y
        """
        x, y = x
        return 0.75*np.exp(-(9*x - 2)**2 / 4 - (9*y - 2)**2 / 4) \
            + 0.75*np.exp(-(9*x + 1)**2 / 49 - (9*y + 1) / 10) \
            + 0.5*np.exp(-(9*x - 7)**2 / 4 - (9*y - 3)**2 / 4) \
            - 0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2) \
            + noise_magnitude * np.random.randn(len(x), len(x))



    def plot(self, x, y, y_tilde):
        """
        Displays two 3-dimensional plots, one of the original data and
        one of the computed best fit.
        """
        y_tilde = np.reshape(y_tilde, np.shape(x[0]))
        #print(f"x: {x.shape} \n y: {y.shape} \n y_tilde: {y_tilde.shape}")
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        surf = ax.plot_surface(x[0], x[1], y, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)


        ax = fig.add_subplot(1, 2, 2, projection='3d')
        surf = ax.plot_surface(x[0], x[1], y_tilde, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()



    def mean_squared_error(self, y, y_tilde):
        if len(y.shape) > 1:
            y = np.ravel(y)
            y_tilde = np.ravel(y_tilde)

        return np.mean((y - y_tilde)**2)



    def r2_score(self, y, y_tilde):
        if len(y.shape) > 1:
            y = np.ravel(y)
            y_tilde = np.ravel(y_tilde)

        return 1 - np.sum((y - y_tilde) ** 2) / np.sum((y - np.mean(y)) ** 2)



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
    parser.add_argument('--cross', help='Use cross validation', action='store_true')
    args = parser.parse_args()

    least_squares = LeastSquares(args.N, args.method, args.noise_magnitude, args.poly_degree)

    x, y = least_squares.init_franke()

    if args.cross:
        least_squares.cross_validation(x, y)
    else:
        design_matrix = least_squares.create_design_matrix(x)
        y_tilde = least_squares.find_fit(y, design_matrix)

    if args.plot:
        least_squares.plot(x, y, y_tilde)
    if args.test:
        least_squares.test_error_analysis(y, y_tilde)
