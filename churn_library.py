"""
Churn prediction excercise for the ML DevOps Engineer nanodegree

Author: Andrea Vitali
Date: Mar 2024
"""


# import libraries
import os
import joblib

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import RocCurveDisplay, classification_report

os.environ["QT_QPA_PLATFORM"] = "offscreen"


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: (str) a path to the csv
    output:
            df: pd.DataFrame
    """
    df = pd.read_csv(pth)
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df.drop("Attrition_Flag", axis=1, inplace=True)
    return df


def perform_eda(df, heatmap_columns, figs_dir, figsize=(20, 10)):
    """
    perform eda on df and save figures to images folder
    input:
            df: (pd.DataFrame) pandas dataframe that contains the data for eda
            heatmap_columns: (list) columns to be used by the heatmap function
            figs_dir: (str) path to the directory where images will be saved
            figsize: (tuple) size of the plots

    output:
            None
    """

    rcParams["figure.figsize"] = figsize
    sns.set_theme(rc={"figure.figsize": figsize})

    path = os.path.join(figs_dir, "churn_distribution.png")
    df["Churn"].hist()
    plt.savefig(path)
    plt.close()

    path = os.path.join(figs_dir, "customer_age_distribution.png")
    df["Customer_Age"].hist()
    plt.savefig(path)
    plt.close()

    path = os.path.join(figs_dir, "marital_stauts_distribution.png")
    df["Marital_Status"].value_counts("normalize").plot(kind="bar")
    plt.savefig(path)
    plt.close()

    path = os.path.join(figs_dir, "total_transaction_distribution.png")
    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)
    plt.savefig(path)
    plt.close()

    path = os.path.join(figs_dir, "heatmap.png")
    sns.heatmap(df.loc[:, heatmap_columns].corr(),
                annot=False, cmap="Dark2_r", linewidths=2)
    plt.savefig(path)
    plt.close()


def encoder_helper(df, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name

    output:
            df: pandas dataframe with new columns for
    """
    column_values_list = []

    for column in category_lst:
        column_values_list = []
        column_groups = df.groupby(column)[response].mean()

        for val in df[column]:
            column_values_list.append(column_groups.loc[val])

        df[f"{column}_{response}"] = column_values_list

    return df


def perform_feature_engineering(
        df,
        keep_cols,
        response,
        test_size,
        random_state=42):
    """
    Transforms raw DatFrame in features and target that can be used for training
    input:
              df: (pd.DataFrame)
              keep_cols: (list) columns to be kept
              response: (str) response name 

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    x = df.loc[:, keep_cols]
    y = df[response]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)

    return (x_train, x_test, y_train, y_test)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds,
                                y_test_preds,
                                model_name,
                                output_path):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds: training predictions
            y_test_preds: test predictions
            model_name: model that produced the predictions
            output_path: path where the figure qill be saved

    output:
             None
    """

    plt.rc("figure", figsize=(5, 5))
    plt.text(0.01, 1.25, str(f"{model_name} Train"), {
             "fontsize": 10}, fontproperties="monospace")
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds)), {
             "fontsize": 10}, fontproperties="monospace")  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str(f"{model_name} Test"), {
             "fontsize": 10}, fontproperties="monospace")
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds)), {
             "fontsize": 10}, fontproperties="monospace")  # approach improved by OP -> monospace!
    plt.axis("off")
    plt.savefig(output_path)
    plt.close()


def feature_importance_plot(model, x_data, output_path):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_path: path to store the figure

    output:
             None
    """

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    # Save figure
    plt.savefig(output_path)
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models

    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """

    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default "lbfgs" fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)
    param_grid = \
        {
            "n_estimators": [200, 500],
            "max_features": ["auto", "sqrt"],
            "max_depth": [4, 5, 100],
            "criterion": ["gini", "entropy"]
        }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    # save models
    joblib.dump(
        cv_rfc.best_estimator_,
        os.path.join(
            ".",
            "models",
            "rfc_model.pkl"))
    joblib.dump(lrc, os.path.join(".", "models", "logistic_model.pkl"))

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_rf,
        y_test_preds_rf,
        "Random Forest",
        os.path.join(
            ".",
            "images",
            "results",
            "rf_results.png"))

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_test_preds_lr,
        "Logistic Regression",
        os.path.join(
            ".",
            "images",
            "results",
            "lr_results.png"))

    feature_importance_plot(cv_rfc, pd.concat([x_train, x_test]), os.path.join(
        ".", "images", "results", "feature_importances.png"))

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(cv_rfc.best_estimator_, x_test, y_test, ax=ax, alpha=0.8)
    RocCurveDisplay.from_estimator(lrc, x_test, y_test, ax=ax, alpha=0.8)
    plt.savefig(os.path.join(".", "images", "results", "roc_curve_result.png"))
    plt.close()


if __name__ == "__main__":

    FIGSIZE = (20, 10)
    EDA_IMAGES_DIR = os.path.join(".", "images", "eda")
    DATA_PATH = os.path.join(".", "data", "bank_data.csv")
    TARGET_COL = "Churn"
    TEST_PERC = 0.3
    SEED = 42

    CAT_COLUMNS = \
        [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category"
        ]

    QUANT_COLUMNS = \
        [
            "Customer_Age",
            "Dependent_count",
            "Months_on_book",
            "Total_Relationship_Count",
            "Months_Inactive_12_mon",
            "Contacts_Count_12_mon",
            "Credit_Limit",
            "Total_Revolving_Bal",
            "Avg_Open_To_Buy",
            "Total_Amt_Chng_Q4_Q1",
            "Total_Trans_Amt",
            "Total_Trans_Ct",
            "Total_Ct_Chng_Q4_Q1",
            "Avg_Utilization_Ratio"
        ]

    KEEP_COLUMNS = \
        [
            "Customer_Age",
            "Dependent_count",
            "Months_on_book",
            "Total_Relationship_Count",
            "Months_Inactive_12_mon",
            "Contacts_Count_12_mon",
            "Credit_Limit",
            "Total_Revolving_Bal",
            "Avg_Open_To_Buy",
            "Total_Amt_Chng_Q4_Q1",
            "Total_Trans_Amt",
            "Total_Trans_Ct",
            "Total_Ct_Chng_Q4_Q1",
            "Avg_Utilization_Ratio",
            "Gender_Churn",
            "Education_Level_Churn",
            "Marital_Status_Churn",
            "Income_Category_Churn",
            "Card_Category_Churn"
        ]

    dataframe = import_data(DATA_PATH)
    perform_eda(dataframe, QUANT_COLUMNS, EDA_IMAGES_DIR, FIGSIZE)
    dataframe = encoder_helper(dataframe, CAT_COLUMNS, TARGET_COL)
    data = perform_feature_engineering(
        dataframe, KEEP_COLUMNS, TARGET_COL, TEST_PERC, SEED)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = data
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
