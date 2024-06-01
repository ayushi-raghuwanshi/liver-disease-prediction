import pandas as pd
from data_preprocessing import load_dataset, balance_liver_data, remove_duplicates, normalize_data

def preprocess_liver_disorder_data(file_path):
    # Load the dataset
    dataset = load_dataset(file_path)
    
    # Balance the dataset
    resampled_dataset = balance_liver_data(dataset)
    
    # Remove duplicates
    resampled_dataset = remove_duplicates(resampled_dataset)
    
    # Separate features and target again after removing duplicates
    X_res = resampled_dataset.drop('CLASS', axis=1)
    y_res = resampled_dataset['CLASS']

    # Normalize data
    X_res = normalize_data(resampled_dataset)

    # # Remove outliers
    # X_res = remove_outliers_zscore(X_res)


    # Combine features and target again
    # final_dataset = pd.concat([X_res, y_res], axis=1)

    #to count total number of class wise values
    # class_counts = final_dataset['CLASS'].value_counts()  o/p: 1- 4115, 0- 4109

    return X_res
