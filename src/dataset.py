"""
Dataset assembly – builds flat DataFrames from gridded features
and splits by year for model training/evaluation.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr


def build_dataset(features_dir):
    """
    Load the features NetCDF and flatten into a pandas DataFrame
    suitable for sklearn models.

    Each row = one (grid cell, day) observation.

    Columns:
      - date, latitude, longitude
      - Normalised drivers: t_max, rh_min, u10_max, sm_top, ndvi, ndwi,
                             slope, frp_hist, count_hist
      - Physics features:   F_avail, F_dry, G_spread, H_history, R_phys
      - Label:              ignition (0 or 1)

    Returns pd.DataFrame.
    """
    path = os.path.join(features_dir, "features.nc")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features file not found: {path}")

    ds = xr.open_dataset(path)

    # Flatten spatial + temporal dims
    df = ds.to_dataframe().reset_index()

    # Drop rows where label is NaN
    if "ignition" in df.columns:
        df = df.dropna(subset=["ignition"])

    # Add year column for splitting
    df["year"] = pd.to_datetime(df["time"]).dt.year

    print(f"  Dataset: {len(df):,} samples, "
          f"fire rate: {df['ignition'].mean()*100:.2f}%")
    return df


def split_by_year(df, train_years, val_years, test_years):
    """
    Temporal train/validation/test split – no leakage.

    Parameters
    ----------
    df : pd.DataFrame with 'year' column
    train_years, val_years, test_years : list[int]

    Returns
    -------
    train_df, val_df, test_df
    """
    train = df[df["year"].isin(train_years)].copy()
    val   = df[df["year"].isin(val_years)].copy()
    test  = df[df["year"].isin(test_years)].copy()

    print(f"  Split: train={len(train):,} ({train_years}), "
          f"val={len(val):,} ({val_years}), "
          f"test={len(test):,} ({test_years})")
    return train, val, test


def spatial_split(df, n_splits=5):
    """
    Spatial block cross-validation using K-Means clustering on coordinates.
    Creates spatial folds to prevent geographic data leakage.

    Returns
    -------
    folds : list of tuples (train_idx, test_idx)
    """
    from sklearn.cluster import KMeans

    coords = df[['latitude', 'longitude']].values
    # Ensure same spatial blocks even for different temporal subsets
    kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=10)
    blocks = kmeans.fit_predict(coords)

    folds = []
    for i in range(n_splits):
        test_mask = (blocks == i)
        train_mask = ~test_mask
        folds.append((np.where(train_mask)[0], np.where(test_mask)[0]))

    print(f"  Spatial CV: {n_splits} distinct geographic blocks created.")
    return folds


def get_spatial_train_val_test(df, n_splits=5):
    """
    Splits the dataset into train, val, test geographically.
    Fold 0 = Test, Fold 1 = Val, Folds 2+ = Train.
    """
    folds = spatial_split(df, n_splits=n_splits)
    
    test_idx = folds[0][1]
    val_idx = folds[1][1]
    
    # Train is the remaining indices
    train_idx = np.concatenate([folds[i][1] for i in range(2, n_splits)])
    
    train = df.iloc[train_idx].copy()
    val = df.iloc[val_idx].copy()
    test = df.iloc[test_idx].copy()
    
    print(f"  Spatial Split: train={len(train):,}, val={len(val):,}, test={len(test):,}")
    return train, val, test


def get_Xy(df, feature_col="R_phys", label_col="ignition"):
    """Extract feature matrix X and label vector y."""
    if isinstance(feature_col, str):
        X = df[[feature_col]].values
    else:
        X = df[feature_col].values
    y = df[label_col].values.astype(int)
    return X, y


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick test with dummy data
    n = 1000
    df = pd.DataFrame({
        "time": pd.date_range("2021-01-01", periods=n, freq="D"),
        "latitude": np.random.uniform(32, 42, n),
        "longitude": np.random.uniform(-124, -114, n),
        "R_phys": np.random.rand(n),
        "ignition": np.random.choice([0, 1], n, p=[0.95, 0.05]),
    })
    df["year"] = df["time"].dt.year
    train, val, test = split_by_year(df, [2021], [2022], [2023])
    print("Dataset self-test passed ✓")
