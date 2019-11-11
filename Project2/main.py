import numpy as np
from neural_network import NeuralNetwork
from data_handling import read_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import seaborn as sbr
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

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

if __name__ == "__main__":
    
    hidden_layers = [50, 50, 50]
    hidden_funcs = ["sigmoid", "sigmoid", "sigmoid"]
    output_func = "softmax"
    eta = 0.056
    lambda_ = 1e-4
    epochs = 100
    
    
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