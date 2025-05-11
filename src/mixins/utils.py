import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Tuple
import re

INT_DTYPES = ["int8", "int16", "int32", "int64"]
FLOAT_DTYPES = ["float", "float16", "float32", "float64"]

def sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sort columns in a dataframe in alphabetical order"""
    return df.reindex(sorted(df.columns), axis=1)

def cats_first_floats_later(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rearanges df columns to have labels first and any floats later
    column order is important for nn
    """
    cats_cols = df.select_dtypes(exclude=FLOAT_DTYPES)
    float_cols = df.select_dtypes(include=FLOAT_DTYPES)

    cats_cols = sort_columns(cats_cols)
    float_cols = sort_columns(float_cols)

    return pd.concat([cats_cols, float_cols], axis=1)

def standardize_float_columns(
    df: pd.DataFrame, scaler: Optional[StandardScaler] = None, ignore_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Standardize float columns so they have mean=0 and std=1, using sklearn's StandardScaler.

    1. Identifies float columns via df.select_dtypes(include=['float']).
    2. Excludes any columns given in ignore_cols.
    3. If no scaler is provided, it creates & fits a new StandardScaler on these columns.
    4. Otherwise, it uses the existing scaler to transform the columns.
    5. Returns a copy of the DataFrame with only the float columns scaled, plus the scaler object.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    scaler : StandardScaler, optional
        Existing scaler to use for transforming. If None, a new StandardScaler will be fit.
    ignore_cols : list of str, optional
        Columns to ignore from scaling.

    Returns
    -------
    scaled_df : pd.DataFrame
        A copy of the original DataFrame but with float columns standardized.
    scaler : StandardScaler
        The fitted scaler (new or existing) for future transforms.
    """
    if ignore_cols is None:
        ignore_cols = []

    # Identify float columns, excluding ignored ones
    float_cols = [col for col in df.select_dtypes(include=["float"]).columns if col not in ignore_cols]

    # Copy DataFrame so original isn't modified
    scaled_df = df.copy()

    # If no scaler is provided, create a new one and fit it
    if scaler is None:
        scaler = StandardScaler()
        scaled_df[float_cols] = scaler.fit_transform(scaled_df[float_cols])
    else:
        # Use the existing scaler to transform these columns
        scaled_df[float_cols] = scaler.transform(scaled_df[float_cols])

    return scaled_df

def is_float(s:str):
    """Checks string for float type"""
    # Pattern explanation:
    # ^[+-]?      : Optional sign (+/-)
    # \d+\.\d+    : At least one digit before and after the decimal point (e.g., "12.3")
    # |\.\d+      : Or a decimal point followed by digits (e.g., ".5")
    # |\d+\.?     : Or digits with an optional trailing decimal point (e.g., "5.")
    # (?:...)$    : Non-capturing group to end the string
    pattern = r'^[+-]?(?:\d+\.\d+|\.\d+|\d+\.?)$'
    return re.fullmatch(pattern, s) is not None

def apply_prefix_to_dtype_dict(base_dtype_dict:dict, prefixes:list, available_columns=None):
    """
    Given a dtype mapping with base column names and a list of prefixes,
    returns a new mapping with prefixed column names. Optionally filters
    by available columns.

    Args:
        base_dtype_dict (dict): base column name -> dtype
        prefixes (list of str): list of prefixes to apply
        available_columns (set or list, optional): if provided, only includes matching columns

    Returns:
        dict: prefixed column name -> dtype
    """
    prefixed_dict = {}
    for base_col, dtype in base_dtype_dict.items():
        for prefix in prefixes:
            full_col = f"{prefix}_{base_col}"
            if available_columns is None or full_col in available_columns:
                prefixed_dict[full_col] = dtype
    return prefixed_dict