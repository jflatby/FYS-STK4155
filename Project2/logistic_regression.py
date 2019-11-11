import numpy as np
import seaborn as sbr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from data_handling import read_data
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
np.random.seed(1337)



def create_design_matrix(features):
    """Function for setting up design matrix and target arrays from the
    data set.

    Parameters:
    dataframe (pandas dataframe): dataframe consisting of credit card default
                                    data.
    prediction_target (str): the target for the logistic regression prediction

    Returns:
    design matrix (nd.array): numpy array representing the design matrix for
                                the prediction
    target (nd.array): numpy array of prediction target values


    """

    onehotencoder = OneHotEncoder(categories='auto')

    design_matrix = ColumnTransformer(
                    [("", onehotencoder, [2, 3]),],
                    remainder='passthrough'
    ).fit_transform(features)

    scaler = StandardScaler(with_mean=False)
    design_matrix = scaler.fit_transform(design_matrix)

    return design_matrix



def cross_entropy(data, target, beta, lambda_=0.0):
    if lambda_:
        value = -np.sum(target*beta - np.log(1 + np.exp(target*beta))) + lambda_*np.linalg.norm(beta)**2
    else:
        value = -np.sum(target*beta - np.log(1 + np.exp(target*beta)))
    return value



def gradient(data, target, beta, lambda_=0):
    p = 1 / (1 + np.exp(-data @ beta))
    if lambda_:
        value = -np.dot(data.T, (target - p)) + 2*lambda_*beta
        return value
    else:
        value = -np.dot(data.T, (target - p))
        return value


def gradient_descent(data, target, beta, iterations, learning_rate, lambda_=0.0, tolerance=1e-6, verbose=True):

    for i in range(iterations):
        grad = gradient(data, target, beta, lambda_)
        #grad = grad / np.mean(grad)
        beta_new = beta - grad * learning_rate
        norm = np.linalg.norm(beta_new - beta)
        beta = beta_new

        if verbose:
            if i % 500 == 0:
                #print(f'beta new: {beta_new}, norm: {norm}')
                print(f'iteration: {i}, eta: {learning_rate}, lambda_: {lambda_}, gradient: {np.linalg.norm(grad)}')

        if (norm < tolerance):
            print(f'Gradient descent converged in {i} iterations\n')
            return beta, norm


    return beta, norm



def probabilities(data, beta):
    t = np.dot(data, beta)
    sigmoid = 1 / (1 + np.exp(-t))
    return sigmoid



def generate_confusion_matrix(targets, predictions, eta):
    # Create confusion matrix and visualize it
    cm = confusion_matrix(targets, predictions)
    sbr.heatmap(pd.DataFrame(cm), annot=True, cmap="viridis", fmt='g')
    plt.title(f'Confusion matrix for $\\eta = {eta}$')
    plt.ylabel('True value')
    plt.xlabel('Predicted value')
    plt.show()



def generate_heatmap(accuracy, x_range, y_range):
    sbr.heatmap(pd.DataFrame(accuracy), annot=True, cmap="viridis", fmt='g')
    plt.title('Grid-search for logistic regression')
    plt.ylabel('Learning rate: $\\eta$')
    plt.xlabel('Regularization Term: $\\lambda$')
    plt.xticks(ticks=np.arange(len(x_range)) + 0.5, labels=x_range)
    plt.yticks(ticks=np.arange(len(y_range)) + 0.5, labels=y_range)
    plt.show()



if __name__=='__main__':
    df = read_data(filtered=True)
    prediction_target='defaultPaymentNextMonth'


    features = df.loc[:, df.columns != prediction_target].values
    targets = df.loc[:, df.columns == prediction_target].values
    design_matrix = create_design_matrix(features)
    data_train, data_test, targets_train, targets_test = train_test_split(design_matrix, targets, test_size=0.2, shuffle=True)


    search_start, search_end, n_points = -6, 1, 8

    learning_rates = np.logspace(search_start, search_end, n_points)
    lambda_values = np.logspace(search_start, search_end, n_points)
    iterations = 10000
    accuracy = np.zeros((len(learning_rates), len(lambda_values)))

    for i, eta in enumerate(learning_rates):
        for j, lmbd in enumerate(lambda_values):
            beta = np.random.uniform(-0.5, 0.5, (data_train.shape[1]))
            beta = np.reshape(beta, (beta.shape[0], 1))
            optimal_beta, norm = gradient_descent(data_train, targets_train, beta, iterations=iterations, learning_rate=eta, lambda_=lmbd)
            probability = probabilities(data_test, optimal_beta)
            predictions = (probability >= 0.5).astype(int)
            accuracy[i, j] = np.mean(predictions == targets_test)
            difference = targets_test - predictions


            print(f'Accurary: {accuracy}, Learning Rate: {eta}, Lambda: {lmbd}')
            print(f'Number Correct: {np.sum(difference == 0)}')

    generate_heatmap(accuracy, learning_rates, lambda_values)
    #generate_confusion_matrix(targets_test, predictions, eta)
