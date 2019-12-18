import numpy as np
from neural_network import NeuralNetwork
from data_handling import read_data, get_franke_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import seaborn as sbr
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from mpl_toolkits.mplot3d import Axes3D

def init_data():
    df = read_data()
    X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
    y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

    onehotencoder = OneHotEncoder(categories="auto", sparse=False)
    X = ColumnTransformer(
                [("", onehotencoder, [2, 3]),],
                remainder="passthrough"
            ).fit_transform(X)
    
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X)
    
    return X, y

def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector

def sklearn_classifier(data_train, targets_train, data_test, eta, hidden_layers, epochs, lmbd):
    #TODO batch size unused for now
    scikit_NN = MLPClassifier(solver='sgd', alpha=lmbd, learning_rate='constant', learning_rate_init=eta, activation='logistic', hidden_layer_sizes=hidden_layers, random_state=1,max_iter=epochs)

    scikit_NN.fit(data_train, targets_train[:, 0])
    y_pred_test = scikit_NN.predict(data_test)
    y_pred_train = scikit_NN.predict(data_train)
    return y_pred_train, y_pred_test

def mean_squared_error(y, y_tilde):
        """
        Returns the means squared error between y and y_tilde
        """
        if len(y.shape) > 1:
            y = np.ravel(y)
            
        y_tilde = np.ravel(y_tilde)

        return np.mean((y - y_tilde)**2)

def regression_test(N, layers, funcs, output, eta, lmbd, epochs):
    
    X, y = get_franke_data(N)
    
    #x = X[0]
    #y = X[1]
    
    X = np.concatenate((np.ravel(X[0]), np.ravel(X[1])), axis=0 )#np.array([x.ravel, y.ravel])
    
    print(X.shape, y.shape)
    
    
    data_train, data_test, targets_train, targets_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    #sm = SMOTE(random_state=42)
    #data_train, targets_train = sm.fit_resample(data_train, targets_train)

    #targets_train_onehot, targets_test_onehot = to_categorical_numpy(targets_train), to_categorical_numpy(targets_test)

    neural_net = NeuralNetwork(data_train, targets_train, layers, funcs, output, eta, lmbd)
    
    neural_net.train(epochs)
    predictions = neural_net.predict_regression(data_train)
    
    #pred_train, sklearn_predictions = sklearn_classifier(data_train, targets_train, data_test, eta, hidden_layers, epochs, lambda_)
    
    #print("----Our Neural Network----")
    print(mean_squared_error(targets_train, predictions))
    #print(np.unique(targets_test, return_counts=True), np.unique(predictions, return_counts=True))
    #print(f"accuracy: {accuracy_score(targets_test, predictions)}")
    #print(f"F1 Score: {f1_score(targets_test, predictions)}")
    
    #print("----SKLearn Neural Network----")
    #print(f"accuracy: {accuracy_score(targets_test, sklearn_predictions)}")
    #print(f"F1 Score: {f1_score(targets_test, sklearn_predictions)}")
    
    return predictions
    
def loop_lmbd_eta(data_train, targets_train, data_test, targets_test, hidden_layers, epochs, activation_funcs, output_func):
    sbr.set()
    
    eta_vals = np.logspace(-7, -4, 4)
    lmbd_vals = np.logspace(-7, -4, 4)
    
    #eta_vals = np.round(np.linspace(0.0001, 0.01, 10), decimals=4) #np.logspace(-4, -2, 6)
    #lmbd_vals = np.round(np.linspace(0.0001, 0.01, 10), decimals=4) #np.logspace(-4, -2, 6)

    train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

    for i in range(len(eta_vals)):
        for j in range(len(lmbd_vals)):
            print("eta value ({} of {}), lmbd value ({} of {})".format(i+1, len(eta_vals), j+1, len(lmbd_vals)))
            DNN = NeuralNetwork(data_train, targets_train, hidden_layers=hidden_layers, learning_rate=eta_vals[i], lambda_=lmbd_vals[j], hidden_activation_functions=activation_funcs, output_activation_function=output_func)
            DNN.train(epochs)
            
            #DNN = NeuralNet(data_train, Y_train_onehot, network_shape[1:-1], hidden_a_func=["sigmoid"])
            #DNN.train(iters=epochs, gamma=eta_vals[i], lmbd=lmbd_vals[j])
            #prediction_train, prediction_test = skullkit_skull(data_train, Y_train_onehot, data_test, eta_vals[i], [50, 50], epochs, batches, lmbd_vals[j])
            
            prediction_train = DNN.predict_regression(data_train)
            prediction_test = DNN.predict_regression(data_test)
            
            train_accuracy[i][j] = mean_squared_error(targets_train, prediction_train)
            test_accuracy[i][j] = mean_squared_error(targets_test, prediction_test)
            
    #print(test_accuracy)
        
    fig, ax = plt.subplots(figsize = (10, 10))
    sbr.heatmap(pd.DataFrame(train_accuracy), annot=True, ax=ax, cmap="viridis", fmt=".4g")
    ax.set_title("Training Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    #plt.yscale("log")
    plt.xticks(ticks=np.arange(len(lmbd_vals)) + 0.5, labels=lmbd_vals)
    plt.yticks(ticks=np.arange(len(eta_vals)) + 0.5, labels=eta_vals)
    plt.savefig("kartong_training_E_{}.png".format(epochs))
    plt.show()

    fig, ax = plt.subplots(figsize = (10, 10))
    sbr.heatmap(pd.DataFrame(test_accuracy), annot=True, ax=ax, cmap="viridis", fmt=".4g")
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    #plt.yscale("log")
    plt.xticks(ticks=np.arange(len(lmbd_vals)) + 0.5, labels=lmbd_vals)
    plt.yticks(ticks=np.arange(len(eta_vals)) + 0.5, labels=eta_vals)
    plt.savefig("kartong_test_E_{}.png".format(epochs))
    plt.show()

if __name__ == "__main__":
    
    hidden_layers = [50, 50, 50]
    hidden_funcs = ["sigmoid", "sigmoid", "sigmoid"]
    output_func = ""
    eta = 0.076
    lambda_ = 1e-4
    epochs = 100
    
    N = 1000
    
    #regression_test(100, hidden_layers, hidden_funcs, output_func, eta, lambda_, epochs)
    
    X, y = get_franke_data(N)
    
    X, y = np.reshape(X, (N*N, 2)), np.reshape(y, (N*N, 1))
    
    data_train, data_test, targets_train, targets_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    
    
    
    loop_lmbd_eta(data_train, targets_train, data_test, targets_test, hidden_layers, epochs, hidden_funcs, output_func)
    
    """
    X, y = init_data()
    
    data_train, data_test, targets_train, targets_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    sm = SMOTE(random_state=42)
    data_train, targets_train = sm.fit_resample(data_train, targets_train)

    targets_train_onehot, targets_test_onehot = to_categorical_numpy(targets_train), to_categorical_numpy(targets_test)

    neural_net = NeuralNetwork(data_train, targets_train_onehot, hidden_layers, hidden_funcs, output_func, eta, lambda_)
    
    neural_net.train(epochs)
    predictions = neural_net.predict(data_test)
    
    pred_train, sklearn_predictions = sklearn_classifier(data_train, targets_train_onehot, data_test, eta, hidden_layers, epochs, lambda_)
    
    print("----Our Neural Network----")
    #print(np.unique(targets_test, return_counts=True), np.unique(predictions, return_counts=True))
    print(f"accuracy: {accuracy_score(targets_test, predictions)}")
    print(f"F1 Score: {f1_score(targets_test, predictions)}")
    
    print("----SKLearn Neural Network----")
    print(f"accuracy: {accuracy_score(targets_test, sklearn_predictions)}")
    print(f"F1 Score: {f1_score(targets_test, sklearn_predictions)}")

    cm = confusion_matrix(targets_test, predictions)
    sbr.heatmap(pd.DataFrame(cm), annot=True, cmap="viridis", fmt="g")
    plt.show()
    
    cm = confusion_matrix(targets_test, sklearn_predictions)
    sbr.heatmap(pd.DataFrame(cm), annot=True, cmap="viridis", fmt="g")
    plt.show()
    #"""