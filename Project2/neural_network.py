import numpy as np
import seaborn as sbr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from data_handling import read_data




class NeuralNetwork:
    def __init__(self, x_data, y_data, network_shape):
        self.x_data = x_data
        self.y_data = y_data
        self.network_shape = network_shape
        self.number_of_layers = len(network_shape)
        self.categories = network_shape[-1]
        self.n_inputs = x_data.shape[0]
        self.n_features = x_data.shape[1]



    def create_bias_and_weights(self):
        self.bias = [0 for i in range(self.number_of_layers-1)]
        self.weights = [0 for i in range(self.number_of_layers-1)]

        for i, l in enumerate(self.network_shape[1:]):
            print(self.network_shape[i], l)
            self.bias[i] = np.zeros(l) + 0.1
            self.weights[i] = np.random.randn(l, self.network_shape[i])

        self.bias = np.array(self.bias)
        self.weights = np.array(self.weights)



    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))


    def feed_forward(self):



        z = [0 for i in range(self.number_of_layers-1)]
        """
        for l in range(self.number_of_layers-1):
            for j in range(self.network_shape[i+1]):
                print(f'l: {l} weights: {self.weights[l][j]}')
                if i == 0:
                    neuron_sum = np.sum(self.weights[l][j] * self.x_data + self.bias[l][j])
                    activation = self.sigmoid(neuron_sum)
                    z[i].append(activation)
                else:
                    neuron_sum = np.sum(self.weights[l][j] * z[l-1][j] + self.bias[l][j])
                    activation = self.sigmoid(neuron_sum)
                    z[i].append(activation)
        #"""

        #"""
        for i, l in enumerate(self.network_shape[1:]):
            print(f'i: {i}')
            if i == 0:
                z[i] = self.sigmoid(np.matmul(self.x_data, self.weights[i].T))
                print(f'z shape: {z[i].shape}\n')
            elif i == number_of_layers-1:
                z[i] =
            else:
                z[i] = self.sigmoid(np.matmul(z[i-1], self.weights[i].T))
                print(f'z shape: {z[i].shape}\n')
        #"""

        print(z)






def batch_splitting(inputs, targets, number_of_batches=15, randomize=True):
    if randomize:
        mask = np.random.shuffle(np.arange(len(inputs)))
        inputs = inputs[mask]
        targets = targets[mask]

    indeces = np.linspace(0, len(inputs), number_of_batches+1, dtype=int)

    for i in range(number_of_batches):
        batch_inputs = inputs[indeces[i]:indeces[i+1]]
        batch_targets = targets[indeces[i]:indeces[i+1]]

        yield batch_inputs, batch_targets




if __name__=='__main__':
    df = read_data()
    features = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
    targets = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values
    data_train, data_test, targets_train, targets_test = train_test_split(features, targets, test_size=0.2, shuffle=True)
    print(f'data train shape: {data_train.shape} ------- target shape: {targets_train.shape}')
    neural_net = NeuralNetwork(data_train, targets_train, [23, 5, 2, 1])
    neural_net.create_bias_and_weights()
    print(f'bias shape: {neural_net.bias.shape} ------- weights shape: {neural_net.weights.shape}\n')
    #print(f'bias: {neural_net.bias} -------\n weights : {neural_net.weights}\n\n\n')

    print('Feedforward: \n')
    neural_net.feed_forward()
