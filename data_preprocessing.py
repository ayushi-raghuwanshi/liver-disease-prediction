import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def balance_liver_data(dataset):
    # Separate features and target
    X = dataset.drop('CLASS', axis=1)
    y = dataset['CLASS']

    # Separate the minority and majority class samples
    minority_class = X[y == 0]
    majority_class = X[y == 1]

    # Function to generate synthetic samples
    def generate_synthetic_samples(minority_samples, n_samples):
        synthetic_samples = []
        n_minority_samples = minority_samples.shape[0]

        for _ in range(n_samples):
            # Randomly choose two different minority samples
            idx1, idx2 = np.random.choice(range(n_minority_samples), size=2, replace=False)
            sample1, sample2 = minority_samples.iloc[idx1], minority_samples.iloc[idx2]

            # Generate a new synthetic sample by linear interpolation
            synthetic_sample = sample1 + np.random.rand() * (sample2 - sample1)
            synthetic_samples.append(synthetic_sample)

        return pd.DataFrame(synthetic_samples, columns=minority_samples.columns)

    # Number of synthetic samples to generate
    n_minority_samples_needed = majority_class.shape[0] - minority_class.shape[0]

    # Generate synthetic samples
    synthetic_samples = generate_synthetic_samples(minority_class, n_minority_samples_needed)

    # Combine the original and synthetic samples
    X_res = pd.concat([X, synthetic_samples], ignore_index=True)
    y_res = pd.concat([y, pd.Series([0] * n_minority_samples_needed)], ignore_index=True)

    # Combine X_res and y_res into a single dataframe
    resampled_dataset = pd.concat([X_res, y_res.rename('CLASS')], axis=1)

    return resampled_dataset
    

def remove_duplicates(df):
    df.drop_duplicates(inplace=True)
    return df

def normalize_data(df):
    scaler = MinMaxScaler()
    columns_to_normalize = df.columns[:-1]  # Exclude the target column 'CLASS'
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df

# def remove_outliers_zscore(df, threshold=3):
#     print(df.head()); exit;
#     z_scores = stats.zscore(df.drop(columns=["CLASS"]))
#     df_no_outliers_zscore = df[(z_scores < threshold).all(axis=1)]
#     return df_no_outliers_zscore

# def remove_outliers_zscore(df, threshold=3.0):
#     z_scores = np.abs(stats.zscore(df))
#     return df[(z_scores < threshold).all(axis=1)]
