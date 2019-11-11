import numpy as np
import seaborn as sbr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from data_handling import read_data
import scikitplot as skplt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import CondensedNearestNeighbour

from sklearn.metrics import accuracy_score, f1_score

from scikitplot.metrics import plot_cumulative_gain

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

plt.style.use("ggplot")
np.random.seed(1337)



class NeuralNetwork:
    def __init__(self, x_data, y_data, network_shape, learning_rate=0.01, epochs=100, batches=50, lmbd=0.001):
        self.x_data_full = x_data
        self.y_data_full = y_data
        self.network_shape = network_shape
        self.number_of_layers = len(network_shape)
        self.categories = network_shape[-1]
        self.n_inputs = x_data.shape[0]
        self.n_features = x_data.shape[1]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batches = batches
        self.lambda_ = lmbd
        self.activation_method = "tanh"
        
        self.create_bias_and_weights()


    def create_bias_and_weights(self):
        self.bias = [0 for i in range(self.number_of_layers-1)]
        self.weights = [0 for i in range(self.number_of_layers-1)]

        for i, l in enumerate(self.network_shape[1:]):
            #print(self.network_shape[i], l)
            self.bias[i] = np.zeros(l) + 0.1
            self.weights[i] = np.random.uniform(size=(self.network_shape[i], l))

        self.bias = np.array(self.bias)
        self.weights = np.array(self.weights)
        

    def activation_derivative(self, x, method):
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

        elif method == 'softmax':
            return 1

        elif method == '':
            return 1
        
    def activation(self, x, method):
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

        elif (method == ''):
            return x

        return t


    def feed_forward(self):

        self.z = [0 for i in range(self.number_of_layers-1)]

        for i, l in enumerate(self.network_shape[1:]):
            if i == 0:
                self.z[i] = self.activation(np.matmul(self.x_data, self.weights[i]) + self.bias[i], self.activation_method)
                
            elif i == len(self.network_shape[1:]) - 1:
                z = np.matmul(self.z[i-1], self.weights[i]) + self.bias[i]
                #print(self.weights[i])
                ex = np.exp(z - np.max(z))
                self.z[i] = ex / np.sum(ex, axis=1, keepdims=True)
                
            else:
                self.z[i] = self.activation(np.matmul(self.z[i-1], self.weights[i]) + self.bias[i], self.activation_method)
            
            #print(self.z[i])

    def predict(self, x, probability = True):
        self.x_data = x
        self.feed_forward()
        if probability:
            return self.z[-1]
        else:
            return np.heaviside(self.z[-1] - 0.5, 0)
        

    def back_propagation(self):
        layer_error = self.z[-1] - self.y_data
        weights_gradients = [0 for i in range(self.number_of_layers-1)]
        bias_gradients = [0 for i in range(self.number_of_layers-1)]
        for l in range(self.number_of_layers-2, -1, -1 ):
            if(l == 0): 
                weights_gradients[l] = np.matmul(self.x_data.T, layer_error)
                bias_gradients[l] = np.sum(layer_error, axis=0)
            
            else:
                weights_gradients[l] = np.matmul(self.z[l-1].T, layer_error)
                bias_gradients[l] = np.sum(layer_error, axis=0)
                layer_error = np.matmul(layer_error, self.weights[l].T) * self.activation_derivative(self.z[l-1], self.activation_method)

            #print(np.mean(layer_error))
            #print(bias_gradients[l])
            #print(weights_gradients[l])
            
            if(self.lambda_ > 0.0):
                weights_gradients[l] += self.lambda_ * self.weights[l]
            
            self.weights[l] -= self.learning_rate * np.array(weights_gradients[l]) #/ len(self.x_data)
            self.bias[l] -= self.learning_rate * np.array(bias_gradients[l]) #/ len(self.x_data)
            

        #self.weights -= self.learning_rate * np.array(weights_gradients)

        #self.bias -= self.learning_rate * np.array(bias_gradients)

    def train(self):
        
        for e in range(self.epochs):
            #print(f"current epoch: {e}")
            for batch_input, batch_target in self.batch_splitting(number_of_batches=self.batches):
                self.x_data = batch_input
                self.y_data = batch_target
                
                self.feed_forward()
                self.back_propagation()
                


    def batch_splitting(self, number_of_batches=15, randomize=True):
        if randomize:
            mask = np.random.shuffle(np.arange(len(self.x_data_full)))
            inputs = self.x_data_full[mask]
            targets = self.y_data_full[mask]
        else:
            inputs = self.x_data_full
            outputs = self.y_data_full
            
        indeces = np.linspace(0, len(inputs[0]), number_of_batches+1, dtype=int)
        
        for i in range(number_of_batches):
            batch_inputs = inputs[0][indeces[i]:indeces[i+1]]
            batch_targets = targets[0][indeces[i]:indeces[i+1]]

            yield batch_inputs, batch_targets
            
            

def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector

def loop_lmbd_eta(data_train, Y_train_onehot, data_test, targets_test):
    sbr.set()
    
    #eta_vals = np.logspace(-7, -4, 4)
    #lmbd_vals = np.logspace(-7, -4, 4)
    
    eta_vals = np.round(np.linspace(0.0001, 0.01, 10), decimals=4) #np.logspace(-4, -2, 6)
    lmbd_vals = np.round(np.linspace(0.0001, 0.01, 10), decimals=4) #np.logspace(-4, -2, 6)

    train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    
    epochs = 30
    batches = 50
    network_shape = [8, 50, 50, 2]

    for i in range(len(eta_vals)):
        for j in range(len(lmbd_vals)):
            print("eta value ({} of {}), lmbd value ({} of {})".format(i+1, len(eta_vals), j+1, len(lmbd_vals)))
            DNN = NeuralNetwork(data_train, Y_train_onehot, network_shape=network_shape, learning_rate=eta_vals[i], lmbd=lmbd_vals[j], epochs=epochs, batches=batches)
            DNN.train()
            
            #DNN = NeuralNet(data_train, Y_train_onehot, network_shape[1:-1], hidden_a_func=["sigmoid"])
            #DNN.train(iters=epochs, gamma=eta_vals[i], lmbd=lmbd_vals[j])
            #prediction_train, prediction_test = skullkit_skull(data_train, Y_train_onehot, data_test, eta_vals[i], [50, 50], epochs, batches, lmbd_vals[j])
            
            prediction_train = DNN.predict(data_train).argmax(axis=1)
            prediction_test = DNN.predict(data_test).argmax(axis=1)
            
            train_accuracy[i][j] = f1_score(targets_train, prediction_train)
            test_accuracy[i][j] = f1_score(targets_test, prediction_test)
            
    #print(test_accuracy)
        
    fig, ax = plt.subplots(figsize = (10, 10))
    sbr.heatmap(pd.DataFrame(train_accuracy), annot=True, ax=ax, cmap="viridis", fmt=".4g")
    ax.set_title("Training Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    #plt.yscale("log")
    plt.xticks(ticks=np.arange(len(lmbd_vals)) + 0.5, labels=lmbd_vals)
    plt.yticks(ticks=np.arange(len(eta_vals)) + 0.5, labels=eta_vals)
    plt.savefig("kartong_training_E_{}_B_{}.png".format(epochs, batches))
    plt.show()

    fig, ax = plt.subplots(figsize = (10, 10))
    sbr.heatmap(pd.DataFrame(test_accuracy), annot=True, ax=ax, cmap="viridis", fmt=".4g")
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    #plt.yscale("log")
    plt.xticks(ticks=np.arange(len(lmbd_vals)) + 0.5, labels=lmbd_vals)
    plt.yticks(ticks=np.arange(len(eta_vals)) + 0.5, labels=eta_vals)
    plt.savefig("kartong_test_E_{}_B_{}.png".format(epochs, batches))
    plt.show()

def skullkit_skull(data_train, targets_train, data_test, eta, hidden_layers, epochs, batches, lmbd):
    scikit_NN = MLPClassifier(solver='sgd', alpha=lmbd, learning_rate='constant', learning_rate_init=eta, activation='logistic', hidden_layer_sizes=hidden_layers, random_state=1,max_iter=epochs, batch_size=150)

    scikit_NN.fit(data_train, targets_train[:, 0])
    y_pred_test = scikit_NN.predict(data_test)
    y_pred_train = scikit_NN.predict(data_train)
    return y_pred_train, y_pred_test

def loop_network_shape(data_train, Y_train_onehot, data_test, targets_test):
    sbr.set()
    
    hidden_layers = np.arange(1, 6)
    neurons_pr_layer = [1, 25, 50, 75, 100]

    train_accuracy = np.zeros((len(hidden_layers), len(neurons_pr_layer)))
    test_accuracy = np.zeros((len(hidden_layers), len(neurons_pr_layer)))
    
    epochs = 30
    batches = 50
    learning_rate = 0.0056
    lambda_ = 0.0078

    for i in range(len(hidden_layers)):
        for j in range(len(neurons_pr_layer)):
            network_shape = [neurons_pr_layer[j] for k in range(hidden_layers[i] + 2)]
            network_shape[0] = 8
            network_shape[-1] = 2
            
            print(f"Network Shape: {network_shape}")
            
            DNN = NeuralNetwork(data_train, Y_train_onehot, network_shape=network_shape, learning_rate=learning_rate, lmbd=lambda_, epochs=epochs, batches=batches)
            DNN.train()
            
            prediction_train = DNN.predict(data_train).argmax(axis=1)
            prediction_test = DNN.predict(data_test).argmax(axis=1)
            
            
            train_accuracy[i][j] = f1_score(targets_train, prediction_train)
            test_accuracy[i][j] = f1_score(targets_test, prediction_test)
            
    #print(test_accuracy)
        
    fig, ax = plt.subplots(figsize = (10, 10))
    sbr.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Training Accuracy")
    ax.set_xlabel("Neurons per layer")
    ax.set_ylabel("Hidden layers")
    plt.yticks(ticks=np.arange(len(hidden_layers)), labels=hidden_layers)
    plt.xticks(ticks=np.arange(len(neurons_pr_layer)), labels=neurons_pr_layer)
    plt.savefig("KAlleF1_training_E_{}_B_{}.png".format(epochs, batches))
    plt.show()

    fig, ax = plt.subplots(figsize = (10, 10))
    sbr.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test Accuracy")
    ax.set_xlabel("Neurons per layer")
    ax.set_ylabel("Hidden layers")
    plt.yticks(ticks=np.arange(len(hidden_layers)), labels=hidden_layers)
    plt.xticks(ticks=np.arange(len(neurons_pr_layer)), labels=neurons_pr_layer)
    plt.savefig("KAlleF1_test_E_{}_B_{}.png".format(epochs, batches))
    plt.show()
    
    
    
    
if __name__=='__main__':
    df = read_data()
    features = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
    targets = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values
    
    #cnn = CondensedNearestNeighbour(random_state=1337)
    #print(f"{features.shape} skalle {np.reshape(targets, (len(targets), )).shape}")
    #features, targets = cnn.fit_resample(features, np.reshape(targets, (len(targets),)))
    #print(f"{features.shape} skalle {targets.shape}")
    
    sm = SMOTE(random_state=42)
    features, targets = sm.fit_resample(features, targets)
    
    data_train, data_test, targets_train, targets_test = train_test_split(features, targets, test_size=0.2, shuffle=True)
    
    #sm = SMOTE(random_state=42)
    #data_train, targets_train = sm.fit_resample(data_train, targets_train)
    
    sc = StandardScaler()
    data_train = sc.fit_transform(data_train)
    data_test = sc.transform(data_test)
    
    Y_train_onehot, Y_test_onehot = to_categorical_numpy(targets_train), to_categorical_numpy(targets_test)
    
    loop_lmbd_eta(data_train, Y_train_onehot, data_test, targets_test)
    
    #loop_network_shape(data_train, Y_train_onehot, data_test, targets_test)
    
    """
    print(f'data train shape: {data_train.shape} ------- target shape: {targets_train.shape}')
    neural_net = NeuralNetwork(data_train, Y_train_onehot, [8, 50, 50, 50, 2], learning_rate=0.0056, lmbd=0.0078, epochs=30, batches=50)
    neural_net.create_bias_and_weights()
    print(f'bias shape: {neural_net.bias.shape} ------- weights shape: {neural_net.weights.shape}\n')
    #print(f'bias: {neural_net.bias} -------\n weights : {neural_net.weights}\n\n\n')
    #print(targets_test.shape)
    print("Starting training...")
    
    neural_net.train()
    
    predictions = neural_net.predict(data_test).argmax(axis=1)
    
    prediction_train, predictions_sklearn = skullkit_skull(data_train, Y_train_onehot, data_test, 0.001, [50, 50, 50], 50, 200, 0.01)
    
    print(Y_test_onehot.shape, predictions.shape)
    print(f"accuracy: {accuracy_score(targets_test, predictions)}")
    print(f"F1 Score: {f1_score(targets_test, predictions)}")
    
    cm = confusion_matrix(targets_test, predictions)
    sbr.heatmap(pd.DataFrame(cm), annot=True, cmap="viridis", fmt="g")
    plt.show()
    
    cm = confusion_matrix(targets_test, predictions_sklearn)
    sbr.heatmap(pd.DataFrame(cm), annot=True, cmap="viridis", fmt="g")
    plt.show()
    
    print("sklearn")
    print(f"accuracy: {accuracy_score(targets_test, predictions_sklearn)}")
    print(f"F1 Score: {f1_score(targets_test, predictions_sklearn)}")
    
    #print(f1_score(targets_test, predictions))
    
    #plt.show()
    #print(np.unique(neural_net.predict(data_test, probability=True)))
    print(np.unique(predictions, return_counts=True))
    #for i in range(len(data_test)):
    #    print(mean_squared_error(targets_test, prediction[i]))
    #"""
        