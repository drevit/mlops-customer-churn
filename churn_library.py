"""
Churn prediction excercise for the ML DevOps Engineer nanodegree

Author: Andrea Vitali
Date: Mar 2024
"""


# import libraries
import os

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

os.environ['QT_QPA_PLATFORM']='offscreen'



def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: (str) a path to the csv
    output:
            df: pd.DataFrame
    '''
    return pd.read_csv(pth)


def perform_eda(df, heatmap_columns, eda_images_dir, figsize=(20,10)):
    '''
    perform eda on df and save figures to images folder
    input:
            df: (pd.DataFrame) pandas dataframe that contains the data for eda
            eda_images_dir: (str) path to the directory where images will be saved
            figsize: (tuple) size of the plots

    output:
            None
    '''

    rcParams['figure.figsize'] = figsize
    sns.set_theme(rc={'figure.figsize':figsize})

    
    df["Churn"] = df["Attrition_Flag"].apply(lambda val: 0 if val == "Existing Customer" else 1)
    df.drop("Attrition_Flag", axis=1, inplace=True)

    path = os.path.join(eda_images_dir, "churn_distribution.png")
    df["Churn"].hist()
    plt.savefig(path)

    path = os.path.join(eda_images_dir, "customer_age_distribution.png")
    df["Customer_Age"].hist()
    plt.savefig(path)

    path = os.path.join(eda_images_dir, "marital_stauts_distribution.png")
    df["Marital_Status"].value_counts("normalize").plot(kind="bar")
    plt.savefig(path)

    path = os.path.join(eda_images_dir, "total_transaction_distribution.png")
    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)
    plt.savefig(path)
    
    path = os.path.join(eda_images_dir, "heatmap.png")
    sns.heatmap(df.loc[:, heatmap_columns].corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig(path)


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    column_values_list = []
    
    for column in category_lst:
        column_values_list = []
        column_groups = df.groupby(column).mean()[response]

        for val in df['Gender']:
            column_values_list.append(column_groups.loc[val])

        df[column] = column_values_list
    
    return df

def perform_feature_engineering(df, keep_cols, response, test_size, random_state=42):
    '''
    Transforms raw DatFrame in features and target that can be used for training
    input:
              df: (pd.DataFrame)
              keep_cols: (list) columns to be kept
              response: (str) response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    X = df.loc[:, keep_cols]
    y = df[response]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=random_state)

    return (X_train, X_test, y_train, y_test)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    

    




def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models

    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)
    param_grid = \
        {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth' : [4,5,100],
            'criterion' :['gini', 'entropy']
        }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    
    feature_importance_plot(model, X_data, output_pth)






if __name__ == "__main__":
    
    figsize = (20,10)
    eda_images_dir = os.path.join(".", "images", "eda")
    data_path = os.path.join(".", "data", "bank_data.csv")
    target_col = "Churn"
    test_size=0.3
    random_state = 42


    
    cat_columns = \
        [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
    
    quant_columns = \
        [
            'Customer_Age',
            'Dependent_count', 
            'Months_on_book',
            'Total_Relationship_Count', 
            'Months_Inactive_12_mon',
            'Contacts_Count_12_mon', 
            'Credit_Limit', 
            'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 
            'Total_Amt_Chng_Q4_Q1', 
            'Total_Trans_Amt',
            'Total_Trans_Ct', 
            'Total_Ct_Chng_Q4_Q1', 
            'Avg_Utilization_Ratio'
        ]
    
    keep_cols = \
        [
            'Customer_Age', 
            'Dependent_count', 
            'Months_on_book',
            'Total_Relationship_Count', 
            'Months_Inactive_12_mon',
            'Contacts_Count_12_mon',
            'Credit_Limit',
            'Total_Revolving_Bal',
            'Avg_Open_To_Buy',
            'Total_Amt_Chng_Q4_Q1',
            'Total_Trans_Amt',
            'Total_Trans_Ct', 
            'Total_Ct_Chng_Q4_Q1', 
            'Avg_Utilization_Ratio',
            'Gender_Churn', 
            'Education_Level_Churn', 
            'Marital_Status_Churn', 
            'Income_Category_Churn', 
            'Card_Category_Churn'
        ]
    
    print("import_data")
    df = import_data(data_path)
    print("perform_eda")
    perform_eda(df, quant_columns, eda_images_dir, figsize=(20,10))
    #df = encoder_helper(df, cat_columns, target_col)

    #data = perform_feature_engineering(df, keep_cols, target_col, test_size, random_state)
    #X_train, X_test, y_train, y_test = data

