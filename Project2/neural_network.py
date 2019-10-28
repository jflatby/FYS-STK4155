import numpy as np
import seaborn as sbr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from data_handling import read_data

np.random.seed(1337)


class NeuralNetwork:
    def __init__(self, x_data, y_data, network_shape, learning_rate=0.01, epochs=100, batches=50):
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
        pos_mask = (x >= 0)
        neg_mask = (x < 0)
        z = np.zeros_like(x)
        z[pos_mask] = np.exp(-x[pos_mask])
        z[neg_mask] = np.exp(x[neg_mask])
        top = np.ones_like(x)
        top[neg_mask] = z[neg_mask]
        return top / (1 + z)


    def feed_forward(self):



        self.z = [0 for i in range(self.number_of_layers-1)]

        for i, l in enumerate(self.network_shape[1:]):
            if i == 0:
                self.z[i] = self.sigmoid(np.matmul(self.x_data, self.weights[i].T) + self.bias[i])
            else:
                self.z[i] = self.sigmoid(np.matmul(self.z[i-1], self.weights[i].T) + self.bias[i])


    def predict(self, x, probability = False):
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
                
                layer_error = np.matmul(layer_error, self.weights[l]) * self.z[l-1] * (1 - self.z[l-1])
            
            self.weights[l] -= self.learning_rate * np.array(weights_gradients[l]).T
            self.bias[l] -= self.learning_rate * np.array(bias_gradients[l])
            
        #self.weights -= self.learning_rate * np.array(weights_gradients).T
        #self.bias -= self.learning_rate * np.array(bias_gradients)

    def train(self):
        
        for e in range(self.epochs):
            print(f"current epoch: {e}")
            for batch_input, batch_target in self.batch_splitting(number_of_batches=neural_net.batches):
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
            
    

if __name__=='__main__':
    df = read_data()
    features = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
    targets = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values
    data_train, data_test, targets_train, targets_test = train_test_split(features, targets, test_size=0.2, shuffle=True)
    print(f'data train shape: {data_train.shape} ------- target shape: {targets_train.shape}')
    neural_net = NeuralNetwork(data_train, targets_train, [23, 50, 50, 50, 1])
    neural_net.create_bias_and_weights()
    print(f'bias shape: {neural_net.bias.shape} ------- weights shape: {neural_net.weights.shape}\n')
    #print(f'bias: {neural_net.bias} -------\n weights : {neural_net.weights}\n\n\n')
    #print(targets_test.shape)
    print("Starting training...")
    
    neural_net.train()
    
    #prediction = neural_net.predict(data_test)
    
    print(np.unique(neural_net.predict(data_test, probability=True)))
    print(np.unique(neural_net.predict(data_test), return_counts=True))
    #for i in range(len(data_test)):
    #    print(mean_squared_error(targets_test, prediction[i]))
        