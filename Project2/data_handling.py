import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import pandas as pd
import seaborn as sbr
import argparse

def read_data(filtered=True):
    """
    column names:

    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
    'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
    'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
    'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    'defaultPaymentNextMonth'
    """


    nanDict = {}
    if filtered:
        df = pd.read_excel('filtered_credit_card_data.xls', header=0, skiprows=0, index_col=0, na_values=nanDict)
    else:
        df = pd.read_excel('default_of_credit_card_clients.xls', header=1, skiprows=0, index_col=0, na_values=nanDict)
        df.rename(index=str, columns={'default payment next month': 'defaultPaymentNextMonth'}, inplace=True)

    return df



def filter_data(df, filename='filtered_credit_card_data.xls'):
    """Removes unnecessary / unwanted values from datasets and saves to new file

    Parameters:
    df (pandas dataframe): dataframe containing dataset of credit card
                                    default data.
    filename (str): filename of file to save to.
    """

    # Drop unnecessary values
    df = df.drop(df[(df.MARRIAGE == 0)].index)
    df = df.drop(df[(df.EDUCATION == 0) &
                    (df.EDUCATION == 5) &
                    (df.EDUCATION == 6)].index)

    df.to_excel(filename)




def generate_scatter(dataframe, category_1, category_2):
    """Generates scatter plot of the two categories supplied. Will label axes
    and supply title based on categories input.

    Parameters:
    dataframe (pandas dataframe): dataframe containing dataset of credit card
                                    default data.

    category_1 (str): First category for comparison
    category_2 (str): Second category for comparison


    """



    plt.scatter(dataframe[category_1], dataframe[category_2], alpha=0.4, edgecolors='w')
    plt.xlabel(category_1)
    plt.ylabel(category_2)
    plt.title(f'{category_2} against {category_1}')
    plt.show()



def generate_jointplot(dataframe, category_1, category_2):
    """Function using seaborn module for plotting a joint plot of two
    categories present in credit card dataset

    Parameters:
    dataframe (pandas dataframe): dataframe containing dataset of credit card
                                    default data.

    category_1 (str): First category for comparison
    category_2 (str): Second category for comparison


    """


    joint_plot = sbr.jointplot(x=f'{category_1}', y=f'{category_2}',
                            data=dataframe, kind='kde', space=0,
                            height=5, ratio=4)
    plt.show()



def generate_histogram(dataframe, category):
    """Function using seaborn module for plotting of histogram for the given
    category present in the credit card dataset

    Parameters:
    dataframe (pandas dataframe): dataframe containing dataset of credit card
                                    default data.

    category (str): First category for comparison


    """


    histogram = sbr.distplot(dataframe[category])
    plt.show()



def generate_correlation_matrix(dataframe):
    f = plt.figure(figsize=(19, 15))
    plt.matshow(dataframe.corr(), fignum=f.number)
    plt.xticks(range(dataframe.shape[1]), dataframe.columns, fontsize=14, rotation=90)
    plt.yticks(range(dataframe.shape[1]), dataframe.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    #plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filtered', help='uses filtered credit card data', action='store_true')
    args = parser.parse_args()
    dataframe = read_data(filtered=args.filtered)


    generate_correlation_matrix(dataframe)
    #generate_scatter(dataframe, 'MARRIAGE', 'AGE')
    #generate_jointplot(dataframe, 'MARRIAGE', 'AGE')
    #generate_histogram(dataframe, 'AGE')