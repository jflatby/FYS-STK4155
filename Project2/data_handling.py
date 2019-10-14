import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import pandas as pd
import seaborn as sbr
import argparse



def read_data(filename):
    """
    column names:

    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
    'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
    'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
    'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    'defaultPaymentNextMonth'
    """

    nanDict = {}
    df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
    df.rename(index=str, columns={'default payment next month': 'defaultPaymentNextMonth'}, inplace=True)

    # Drop unnecessary values
    df = df.drop(df[(df.MARRIAGE == 0)].index)
    df = df.drop(df[(df.EDUCATION == 0) |
                    (df.EDUCATION == 5) |
                    (df.EDUCATION == 6)].index)

    return df



def generate_scatter(dataframe, category_1, category_2):
    plt.scatter(dataframe[category_1], dataframe[category_2], alpha=0.4, edgecolors='w')
    plt.xlabel(category_1)
    plt.ylabel(category_2)
    plt.title(f'{category_2} against {category_1}')
    plt.show()



def generate_jointplot(dataframe, category_1, category_2):
    joint_plot = sbr.jointplot(x=f'{category_1}', y=f'{category_2}',
                            data=dataframe, kind='kde', space=0,
                            height=5, ratio=4)
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    dataframe = read_data('default of credit card clients.xls')
    dataframe.to_excel('filtered credit card data.xls')
    #values, amount = np.unique(dataframe['EDUCATION'], return_counts=True)
    #print(values, amount)


    #generate_scatter(dataframe, 'LIMIT_BAL', 'AGE')
    #generate_jointplot(dataframe, 'LIMIT_BAL', 'AGE')
