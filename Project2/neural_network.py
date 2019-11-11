import numpy as np

np.random.seed(1337)

class NeuralNetwork:
    
    def __init__(self, X, y, hidden_layers, hidden_activation_functions, output_activation_function, learning_rate, lambda_):
        self.X = X
        self.hidden_layers = hidden_layers
        self.h_activation_funcs = hidden_activation_functions
        self.o_activation_func = output_activation_function
        self.categories = y.shape[1]
        self.eta = learning_rate
        self.lmbd = lambda_
        
        self.y = y
        
        self.initialize()
           
    def initialize(self):
        """
        Initialize all arrays of values based on the amount of hidden layers, 
        neurons in each layer, number of inputs and number of categories.
        """
        self.a = [0 for i in range(len(self.hidden_layers) + 2)]
        self.z = [0 for i in range(len(self.hidden_layers) + 1)]
        self.w = [0 for i in range(len(self.hidden_layers) + 1)]
        self.b = [0 for i in range(len(self.hidden_layers) + 1)]
    
        #Data in input layer
        self.a[0] = self.X
        
        #Weights between input layer and first hidden layer
        self.w[0] = np.random.uniform(size=(self.X.shape[1], self.hidden_layers[0]))
        
        #Hidden layers
        for l, neurons in enumerate(self.hidden_layers):
            self.a[l+1] = np.zeros(neurons)
            self.z[l+1] = np.zeros(neurons)
            self.b[l] = np.zeros(neurons) + 0.01
            
            if l > 0:
                self.w[l] = np.random.uniform(-1, 1, size=(self.hidden_layers[l-1], neurons))
                
        #Last layer
        self.a[-1] = np.zeros(self.categories)
        self.z[-1] = np.zeros(self.categories)
        self.b[-1] = np.zeros(self.categories) + 0.01
        self.w[-1] = np.random.uniform(-1, 1, size=(self.hidden_layers[-1], self.categories))
        
    def feed_forward(self):
        """
        Feed forward through the network, starting at the first layer and
        applying the current biases and weights along the way
        """
        #Feed forward through each hidden layer
        for l in range(1, len(self.hidden_layers) + 1):
            self.z[l] = np.matmul(self.a[l-1], self.w[l-1]) + self.b[l-1]
            self.a[l] = self.activation(self.z[l], self.h_activation_funcs[l-1])
            
        #Last layer
        self.z[-1] = np.matmul(self.a[-2], self.w[-1]) + self.b[-1]
        self.a[-1] = self.activation(self.z[-1], self.o_activation_func)
        
    def gradients(self, layer_error, l):
        """
        Function for calculating the gradients for the biases and weights,
        given the layer_error and current layer
        """
        b_gradient = np.sum(layer_error, axis=0)
        w_gradient = np.matmul(self.a[l].T, layer_error)
        #print(layer_error)
        if self.lmbd > 0:
            w_gradient += self.lmbd * self.w[l]
        
        return b_gradient, w_gradient
        
    def back_propagation(self):
        """
        Use gradient descent to tune the biases and weights
        """
        #Output layer
        layer_error = self.a[-1] - self.y
        
        b_gradient, w_gradient = self.gradients(layer_error, len(self.hidden_layers))

        self.b[-1] -= self.eta * b_gradient / len(self.X)
        self.w[-1] -= self.eta * w_gradient / len(self.X)
        
        #Loop backwards through rest of layers
        for l in range(len(self.hidden_layers), 0, -1):
            layer_error = np.matmul(layer_error, self.w[l].T) * self.activation_derivative(self.a[l], self.h_activation_funcs[l-1])
            
            b_gradient, w_gradient = self.gradients(layer_error, l-1)
            self.b[l-1] -= self.eta * b_gradient / len(self.X)
            self.w[l-1] -= self.eta * w_gradient / len(self.X)
            
    def train(self, epochs):
        """
        Feed forward and back propaagate the specified number of times.
        Batch splitting for stochastic gradient descent would also happen here,
        however it is not included in this class. see old_neural_network.py
        """
        for i in range(epochs):
            self.feed_forward()
            self.back_propagation()
        
    def predict(self, X):
        """
        Use the current weights and biases to make a prediction on the dataset X
        by inserting into the first layer and feeding forward
        """
        self.a[0] = X
        self.feed_forward()
        
        y_pred = np.argmax(self.a[-1], axis=1)
        
        return y_pred
    
    def activation(self, x, method):
        """
        Activation functions
        """
        if (method == 'sigmoid'):
            t = 1./(1 + np.exp(-x))

        elif (method == 'softmax'):
            if len(x.shape) > 1:
                exp_term = np.exp(x)
                t = exp_term / np.sum(exp_term, axis=1, keepdims=True)
            else:
                exp_term = np.exp(x)
                t = exp_term / np.sum(exp_term)

        elif (method == 'tanh'):
            t = np.tanh(x)

        elif (method == 'relu'):
            neg = np.where(x < 0)
            x[neg] = 0
            t = x
        else:
            t = 1

        return t
            
    def activation_derivative(self, x, method):
        """
        Derivative of activation functions
        """
        if method == 'sigmoid':
            return x*(1 - x)

        elif method == 'tanh':
            return 1 - x**2

        elif method == 'relu':
            pos = np.where(x > 0)
            neg = np.where(x <= 0)
            x[pos] = 1
            x[neg] = 0
            return x

        else:
            return 1