import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.linalg as scl
import argparse
from imageio import imread
import time
plt.style.use("ggplot")

def frankes_function(x, noise_magnitude = 0.1):
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

class Regression:
    def __init__(self, x, y, regression_method, poly_degree, lambda_ = 0.001):
        """
        This class takes a dataset x, y, a regression method (ols, ridge, lasso), a polynomial degree to use for the fit and
        optionally a lambda-value in the case of ridge or lasso regression.
        """
        self.y = y
        self.regression_method = regression_method
        self.lambda_ = lambda_
        self.poly_degree = poly_degree
        self.X = self.create_design_matrix(x)
       
    def create_design_matrix(self, x):
        """
        Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
        Input is x=[x, y] mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
        """
        x, y = x
        if len(x.shape) > 1:
            x = np.ravel(x)
        if len(y.shape) > 1:
            y = np.ravel(y)
            
        N = len(x)
        l = int((self.poly_degree+1)*(self.poly_degree+2)/2)		# Number of elements in beta
        X = np.ones((N, l))

        for i in range(1, self.poly_degree+1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                X[:, q+k] = x**(i-k) * y**k

        return X
      
    def find_fit(self, design_matrix, data):
        """
        Use the selected method to find the best fit for the data.

        Returns:
            beta-coefficients for the best fit using the selected method and properties.
        """
        if len(data.shape) > 1:
            data = np.ravel(data)

        if self.regression_method == 'ols':
            beta = self.ordinary_least_squares(design_matrix, data)
        elif self.regression_method == 'ridge':
            beta = self.ridge_regression(design_matrix, data)
        elif self.regression_method == 'lasso':
            beta = self.lasso_regression(design_matrix, data)

        else:
            print('invalid method.')
            return
        
        return beta
        
    def ordinary_least_squares(self, X, y):
        """
        Performs ordinary least squares using np.linalg.lstsq
        SVD does also work but was never fully implemented.
        
        Returns:
            array of beta-values
        """
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        #u, s, v = scl.svd(X)
        #beta = v.T @ scl.pinv(scl.diagsvd(s, u.shape[0], v.shape[0])) @ u.T @ y
        
        return beta
        
    def ridge_regression(self, X, y):
        """
        Performs ridge regression on the given data
        
        Returns:
            Array of beta-values
        """
        beta = np.dot(np.linalg.inv(np.dot(np.transpose(X), X) + self.lambda_ * np.eye(X.shape[1])), np.dot(np.transpose(X),y))

        return beta

    def lasso_regression(self, X, y):
        """
        Performs lasso regression on the given data using sklearns Lasso
        class.
        
        Returns:
            Array of beta-values
        """
        model = linear_model.Lasso(fit_intercept=True, max_iter=1000000, alpha=self.lambda_)
        model.fit(X, y)
        beta = model.coef_
        beta[0] = model.intercept_
        return beta
   
    def k_fold(self, x, y, k=5):
        """
        Takes x and y data, loops through k times and splits the 
        dataset into different training and test sets each time,
        and produces a prediction on the test set using beta-values
        "learned" from the training set. 
        
        Returns:
            Mean Squared Error, Bias, Variance, R^2 score
        """
        kfold = KFold(n_splits = k,shuffle=True,random_state=5)
        y = y.ravel()
        
        y_trainz, y_testz = train_test_split(y, test_size=1./k)
        array_size_thingy=len(y_testz)
        
        ypred = np.empty((array_size_thingy, k))
        ytest = np.empty((array_size_thingy, k))
        j=0
        
        xx, yy = x
        x = np.array([xx.ravel(), yy.ravel()])
        
        #y_no_noise = frankes_function(x, noise_magnitude=0).ravel()
        
        for train_inds,test_inds in kfold.split(y):
            x_train = x[:, train_inds]
            y_train = y[train_inds]
            x_test = x[:, test_inds]
            y_test = y[test_inds]#y_no_noise[test_inds]
            
            X_train = self.create_design_matrix(x_train)
            beta = self.find_fit(X_train, y_train)
            X_test = self.create_design_matrix(x_test)
            #print(X_test.shape)
            ypred[:, j] = np.dot(X_test, beta)
            ytest[:, j] = y_test
            
            j+=1
            
        
        error = np.mean( np.mean((ytest - ypred)**2, axis=1, keepdims=True) )
        bias = np.mean( (ytest - np.mean(ypred, axis=1, keepdims=True))**2 )
        variance = np.mean( np.var(ypred, axis=1, keepdims=True) )
        r2_score = self.r2_score(ytest, ypred)
        
        return error, bias, variance, r2_score
    
    def mean_squared_error(self, y, y_tilde):
        """
        Returns the means squared error between y and y_tilde
        """
        if len(y.shape) > 1:
            y = np.ravel(y)
            
        y_tilde = np.ravel(y_tilde)

        return np.mean((y - y_tilde)**2)

    def r2_score(self, y, y_tilde):
        """
        Computes R^2 score, indicating how well y_tilde fits y
        """
        if len(y.shape) > 1:
            y = np.ravel(y)
            y_tilde = np.ravel(y_tilde)

        return 1 - np.sum((y - y_tilde) ** 2) / np.sum((y - np.mean(y)) ** 2)

    def confidence(self, beta, X, confidence=1.96):
        """
        Computes confidence intervals of the beta coefficients using the chosen
        confidence value.
        
        Returns:
            betamin and betamax
        """
        weight = np.sqrt( np.diag( np.linalg.inv( X.T @ X ) ) )*confidence
        betamin = beta - weight
        betamax = beta + weight
        return betamin, betamax
            
    def plot_3D(self, x, y, y_tilde):
        """
        Displays two 3-dimensional plots, one of the original data and
        one of the computed best fit.
        """
        y_tilde = np.reshape(y_tilde, np.shape(x[0]))
        
        #print(f"x: {x.shape} \n y: {y.shape} \n y_tilde: {y_tilde.shape}")
        fig = plt.figure(figsize=(20, 9))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_title('Data')
        surf = ax.plot_surface(x[0], x[1], y, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()
        fig = plt.figure(figsize=(20, 9))

        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_title(f'Regression Fit ({self.regression_method.capitalize()})')
        surf = ax.plot_surface(x[0], x[1], y_tilde, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        fig.tight_layout()

        plt.show()
        
    def plot_contour(self, x, y, y_tilde):
        """
        Make a contour plot of the data
        """
        y_tilde = np.reshape(y_tilde, np.shape(x[0]))

        skalle = plt.imshow(y_tilde, extent=[0, 1, 0, 1], origin='lower', alpha=1.0, cmap='gist_earth')
        plt.colorbar(skalle)
        plt.grid(False)
        plt.show()
        
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
        

def test_confidence(N, regression_method="ols", poly_degree=5):
    """
    Plot the confidence intervals on the different beta-values
    created by the regression fit.
    """
    x = np.sort(np.random.rand(100))
    y = np.sort(np.random.rand(100))
    x, y = np.meshgrid(x, y)
    
    x = np.array([x, y])
    
    regr = Regression(x, y, regression_method, poly_degree)
    
    y = frankes_function(x)
    
    X = regr.create_design_matrix(x)
    beta = regr.find_fit(X, y)
        
    y_tilde = X @ beta
    
    beta_min, beta_max = regr.confidence(beta, X, 1.96)
    skalle = beta_max - beta_min
    
    plt.scatter(range(len(beta)), beta)
    plt.errorbar(range(len(beta)), beta, yerr=np.array(skalle), lw=1, fmt="none", capsize=3)
    plt.xlabel("Range of $\\beta$'s")
    plt.ylabel("$\\beta$ values")
    plt.legend(["Betas", "Confidence intervals"])
    plt.show()
    

def loop_over_poly(start, end, step, regression_method="ols"):
    """
    Loop over the polynomial degree and perform the chosen regression
    a given amount of times, and plot the resulting error values(from the kfold function)
    against model complexity.
    """
    degrees = np.arange(start, end, step)
    
    x = np.sort(np.random.rand(100))
    y = np.sort(np.random.rand(100))
    x, y = np.meshgrid(x, y)
    
    x = np.array([x, y])
    
    y = frankes_function(x)
    
    y_no_noise = frankes_function(x, noise_magnitude=0)
    
    error = []
    r2_score = []
    bias = []
    variance = []
    for deg in degrees:
        regr = Regression(x, y, regression_method, deg)
        print(f"Degree: {deg}")
        
        err, bi, var, r2 = regr.k_fold(x, y)
        
        r2_score.append(np.mean(r2))
        error.append(np.mean(err))
        bias.append(np.mean(bi))
        variance.append(np.mean(var))
        
    ## Plot MSE og R2
    """     
    plt.plot(degrees, error)
    plt.title("Mean Squared Error(Ridge)")
    plt.xlabel("Model complexity (Polynomial degree)")
    plt.ylabel("MSE")
    plt.show()
    plt.plot(degrees, r2_score)
    plt.title("$R^2 Score$(Ridge)")
    plt.xlabel("Model complexity (Polynomial degree)")
    plt.ylabel("$R^2$")
    plt.show()
    #"""
    
    plt.figure()
    plt.plot(degrees, error,label='MSE')
    plt.plot(degrees, bias,label='Bias^2')
    plt.plot(degrees, variance,label='Var')

    plt.xlabel("Model Complexity")
    #plt.ylabel('MSE, Bias, Variance')
    plt.legend()
    plt.show()
        
    #return error, bias, variance, r2
    
     
    ## plot mse, bias, variance
    """   
    plt.figure()
    plt.plot(degrees, error,label='MSE')
    plt.plot(degrees, bias,label='Bias^2')
    plt.plot(degrees, variance,label='Var')

    plt.xlabel("Budeie")
    plt.ylabel('BudError')
    plt.legend()
    plt.show()
    #"""
    
    
def loop_poly_and_lambda(x, y, regression_method="ridge", degree_end=15):
    """
    Loop over polynomial degree just like the above function(These should have been made so that
    this function uses the above one..) 
    Also loops over different degrees of lambda and plot all of them as their own line vs model complexity.
    """
    N_deg = degree_end
    
    degrees = np.arange(1, N_deg+1, 1)
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]
    N_lambda = len(lambdas)
    
    #y_no_noise = frankes_function(x, noise_magnitude=0)
    
    error = np.zeros((N_deg, N_lambda))
    r2_score = np.zeros((N_deg, N_lambda))
    bias = np.zeros((N_deg, N_lambda))
    variance = np.zeros((N_deg, N_lambda))
    for i in range(len(degrees)):
        print(f"Degree: {i+1}")
        for j in range(len(lambdas)):
            regr = Regression(x, y, regression_method, degrees[i], lambdas[j])
            
            err, bi, var, r2 = regr.k_fold(x, y)
            
            r2_score[i, j] = np.mean(r2)
            error[i, j] = np.mean(err)
            bias[i, j] = np.mean(bi)
            variance[i, j] = np.mean(var)
            
    
    for k in range(len(lambdas)):
        plt.plot(degrees, error[:, k])
        
    plt.title("Mean Squared Error(Ridge)")
    plt.xlabel("Model Complexity(Polynomial degree)")
    plt.ylabel("Mean Squared Error")
    plt.legend(lambdas)
        
    plt.show()
    
    for k in range(len(lambdas)):
        plt.plot(degrees, r2_score[:, k])
    
    plt.title("$R^2$ Score(Ridge)")
    plt.xlabel("Model Complexity(Polynomial degree)")
    plt.ylabel("$R^2$ Score")
    plt.legend(lambdas)
        
    plt.show()
    
    for k in range(len(lambdas)):
        plt.plot(degrees, bias[:, k])
        
    plt.title("Bias(Ridge)")
    plt.xlabel("Model Complexity(Polynomial degree)")
    plt.ylabel("Bias")
    plt.legend(lambdas)
    
    plt.show()
    
    for k in range(len(lambdas)):
        plt.plot(degrees, variance[:, k])
        
    plt.title("Variance(Ridge)")
    plt.xlabel("Model Complexity(Polynomial degree)")
    plt.ylabel("Variance")
    plt.legend(lambdas)
        
        
    plt.show()
    
    #return error, bias, variance, r2_score

def read_tif_file(file):
    data = imread(file)
    print(data.shape)
    for i in range(8):
        data = np.delete(data, slice(None, None, 3), axis=0)
        data = np.delete(data, slice(None, None, 3), axis=1)
        
    print(data.shape)
        
    n, m = data.shape
    
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, m)
    
    x, y = np.meshgrid(x, y)
    
    return x, y, data

if __name__ == "__main__":
    time_start = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='method', type=str, help='Regression method', default='ols')
    parser.add_argument('-degree', dest='poly_degree', type=int, help='Polynomial degree for design matrix', default=5)
    #parser.add_argument('-noise', dest='noise_magnitude', type=float, help='Magnitude for the noise added to Frankes function', default=0.01)
    parser.add_argument('-l', dest='lambda_', type=float, help='Pentalty term for ridge and lasso regression', default=0.001)
    parser.add_argument('--test', help='Runs test functions', action='store_true')
    parser.add_argument('--plot', help='Plots the data and the fit side by side', action='store_true')
    parser.add_argument('--terrain', help='Use terrain data instead of Frankes Function', action='store_true')
    args = parser.parse_args()
    
    x, y, z = read_tif_file("grand_canyon.tif")
    x = np.array([x, y])
    y = z
    
    #loop_poly_and_lambda(x, y, args.method, degree_end=15)
    
    #"""
    if (args.terrain):
        x, y, data = read_tif_file("grand_canyon.tif")
        x = np.array([x, y])
        y = data/np.max(data)
        
        regr = Regression(x, y, args.method, args.poly_degree, args.lambda_)
        
        """ 
        X = regr.create_design_matrix(x)
        beta = regr.find_fit(X, y)
        y_tilde = X @ beta
        """
        
        err, bi, var, r2 = regr.k_fold(x, y)
        
        print(f"mse: {err:.4f} --- bias: {bi:.4f} --- variance: {var:.4f} --- r2: {r2:.4f}")
        
    else:
        x = np.sort(np.random.rand(100))
        y = np.sort(np.random.rand(100))
        x, y = np.meshgrid(x, y)
    
        x = np.array([x, y])
        
        y = frankes_function(x)
        
        regr = Regression(x, y, args.method, args.poly_degree, args.lambda_)
        
        X = regr.create_design_matrix(x)
        beta = regr.find_fit(X, y)
        y_tilde = X @ beta
        
        
        
        
    if(args.plot):
        regr.plot_3D(x, y, y_tilde)
        #regr.plot_contour(x, y, y_tilde)
        
    if args.test:
        regr.test_error_analysis(y, y_tilde) 
        
    print(f"Computation time: {time.time() - time_start}")
    
    
    #print(np.array(error).shape, np.array(bias).shape, np.array(variance).shape, np.array(r2_score).shape)
    #loop_over_poly(1, 30, 1, args.method)
    #test_confidence(100)
    
    #"""