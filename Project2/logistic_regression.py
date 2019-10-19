import numpy as np
import seaborn as sbr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from data_handling import read_data





def create_design_matrix(dataframe, prediction_target='defaultPaymentNextMonth'):
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


    features = dataframe.loc[:, dataframe.columns != prediction_target].values
    target = dataframe.loc[:, dataframe.columns == prediction_target].values

    onehotencoder = OneHotEncoder(categories='auto')

    design_matrix = ColumnTransformer(
                    [("", onehotencoder, [3]),],
                    remainder='passthrough'
    ).fit_transform(features)

    return design_matrix, target



def cross_entropy(design_matrix, beta, target):
    p = np.dot(design_matrix, beta)
    loss = -np.sum(target*p - np.log(1 + np.exp(p)))

    return loss


if __name__=='__main__':
    df = read_data(filtered=True)
