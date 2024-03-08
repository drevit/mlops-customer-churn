"""
Script for testing churn_library.py with pytest

Author: Andrea Vitali
Date: Mar 2024
"""
import os
import logging
# import churn_library_solution as cls
import pytest
from churn_library import import_data, perform_eda, encoder_helper,\
      perform_feature_engineering, train_models


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(scope="module")
def heatmap_columns():
    """
    fixture for heatmap testing
    """
    return [
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
        "Avg_Utilization_Ratio"]


@pytest.fixture(scope="module")
def categorical_columns():
    """
    fixture for categorical columns testing
    """
    return [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category"]


@pytest.fixture(scope="module")
def keep_columns():
    """
    fixture for testing columns to be kept
    """
    return [
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
        "Card_Category_Churn"]


@pytest.fixture(scope="module")
def df_path():
    """
    fixture for importing data
    """
    return os.path.join(".", "data", "bank_data.csv")


@pytest.fixture(scope="module")
def eda_images_dirpath():
    """
    fixture for checking eda images path
    """
    return os.path.join(".", "images", "eda")


@pytest.fixture(scope="module")
def rfc_model_path():
    """
    fixture for checking rfc_model path
    """
    return os.path.join(".", "models", "rfc_model.pkl")


@pytest.fixture(scope="module")
def lrc_model_path():
    """
    fixture for checking lrc_model path
    """
    return os.path.join(".", "models", "logistic_model.pkl")


@pytest.fixture(scope="module")
def roc_curve_path():
    """
    fixture for checking roc curve figure path
    """
    return os.path.join(".", "images", "results", "roc_curve_result.png")


@pytest.fixture(scope="module")
def rf_report_path():
    """
    fixture for checking rf_report figure path
    """
    return os.path.join(".", "images", "results", "rf_results.png")


@pytest.fixture(scope="module")
def lr_report_path():
    """
    fixture for checking lr_report figure path
    """
    return os.path.join(".", "images", "results", "lr_results.png")


@pytest.fixture(scope="module")
def feature_importance_path():
    """
    fixture for checking feature importance figure path
    """
    return os.path.join(".", "images", "results", "feature_importances.png")


@pytest.fixture(scope="module")
def target_column():
    """
    fixture for target column
    """
    return os.path.join("Churn")


def test_import(df_path):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data(df_path)
        logging.info("SUCCESS: Testing import_data csv loading")
    except FileNotFoundError as err:
        logging.error("ERROR: Testing import_eda: the file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("SUCCESS: Testing import_data dataframe's dimensions")
    except AssertionError as err:
        logging.error(
            "ERROR: Testing import_data: the file doesn't appear to have rows or columns")
        raise err


def test_eda(df_path, heatmap_columns, eda_images_dirpath):
    '''
    test perform eda function
    '''
    df = import_data(df_path)

    try:
        perform_eda(df, heatmap_columns, eda_images_dirpath, (20, 10))
        assert os.path.isfile(
            os.path.join(
                eda_images_dirpath,
                "churn_distribution.png"))
        assert os.path.isfile(
            os.path.join(
                eda_images_dirpath,
                "customer_age_distribution.png"))
        assert os.path.isfile(
            os.path.join(
                eda_images_dirpath,
                "marital_stauts_distribution.png"))
        assert os.path.isfile(
            os.path.join(
                eda_images_dirpath,
                "total_transaction_distribution.png"))
        assert os.path.isfile(os.path.join(eda_images_dirpath, "heatmap.png"))
        logging.info("SUCCESS: Testing eda images path")
    except AssertionError:
        logging.error("ERROR: Testing eda images path: file wasn't found")


def test_encoder_helper(df_path, categorical_columns, target_column):
    '''
    test encoder helper
    '''
    df = import_data(df_path)

    try:
        df = encoder_helper(df, categorical_columns, target_column)
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("SUCCESS: Testing encoder_helper dataframe's dimensions")
    except AssertionError:
        logging.info(
            "ERROR: Testing encoder_helper: : the file doesn't appear to have rows or columns")


def test_perform_feature_engineering(
        df_path,
        categorical_columns,
        target_column,
        keep_columns):
    '''
    test perform_feature_engineering
    '''
    df = import_data(df_path)
    df = encoder_helper(df, categorical_columns, target_column)

    try:
        data = perform_feature_engineering(
            df, keep_columns, target_column, 0.3)
        assert len(data) == 4
        logging.info(
            "SUCCESS: Testing perform_feature_engineering output dimensions")
    except AssertionError:
        logging.info(
            f"ERROR: Testing perform_feature_engineering: unexpected output dimension (found: {
                len(data)}, expected: 4)")

    try:
        for arr in data:
            assert arr.shape[0] > 0
        logging.info(
            "SUCCESS: Testing perform_feature_engineering arrays dimension")
    except AssertionError:
        logging.info(
            "ERROR: Testing perform_feature_engineering: unexpected array output dimension")


def test_train_models(
        df_path,
        categorical_columns,
        target_column,
        keep_columns,
        rfc_model_path,
        lrc_model_path,
        roc_curve_path,
        lr_report_path,
        rf_report_path,
        feature_importance_path):
    '''
    test train_models
    '''
    df = import_data(df_path)
    df = encoder_helper(df, categorical_columns, target_column)
    data = perform_feature_engineering(df, keep_columns, target_column, 0.3)
    x_train, x_test, y_train, y_test = data
    train_models(x_train, x_test, y_train, y_test)

    try:
        os.path.isfile(rfc_model_path)
        os.path.isfile(lrc_model_path)
        logging.info("SUCCESS: Testing train_models models' pickle path")
    except AssertionError:
        logging.info(
            "ERROR: Testing train_models: models' files weren't found")

    try:
        assert os.path.isfile(roc_curve_path)
        assert os.path.isfile(lr_report_path)
        assert os.path.isfile(rf_report_path)
        assert os.path.isfile(feature_importance_path)

        logging.info("SUCCESS: Testing train_models figures path")
    except AssertionError:
        logging.info(
            "ERROR: Testing train_models: figures files weren't found")
