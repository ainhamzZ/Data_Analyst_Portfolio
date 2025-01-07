import os
import pandas as pd
import argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder


def clean_data(df):

    # Example Cleaning Steps
    # 1. Fill missing values
    # Iterate through numerical columns
    
    input_cols_mean = input("Please indicate numerical colums to impute mean separated by spaces: ")
    col_impute_mean = input_cols_mean.split()
    for col in df[col_impute_mean]:
        if df[col].isnull().any():  # Check if the column has any missing values
            df[col].fillna(df[col].mean(), inplace=True)  # Impute with mean
            
    # Iterate through categorical columns
    input_cols_mode = input("Please indicate categorical colums to impute mode separated by spaces: ")
    col_impute_mode = input_cols_mode.split()
    label_encoder = LabelEncoder()
    for col2 in df[col_impute_mode]:
        if df[col2].isnull().any():  # Check if the column has any missing values
            df[col2].fillna(df[col2].mode()[0], inplace=True)  # Impute with mode
            df[col2] = label_encoder.fit_transform(df[col2])  # Apply label encoding
        else:
            df[col2] = label_encoder.fit_transform(df[col2])  # Apply label encoding
        
    return df


if __name__ == "__main__":
    # Set up argument parsing   
    parser = argparse.ArgumentParser(description="Clean and drop specified columns from a DataFrame")
    parser.add_argument(
    "--columns", 
    required=True, 
    help="Comma-separated column names to drop, e.g., 'Age,City'"
)
    parser.add_argument(
    "--input", 
    required=True, 
    help="Path to the CSV file to clean"
)
    parser.add_argument(
    "--output", 
    required=True, 
    help="Path to save the cleaned CSV file"
)
   
    # Parse arguments
    args = parser.parse_args()

    # Load the DataFrame
    print(f"Attempting to load file from: {args.input}")
    df = pd.read_csv(args.input)
    
    # Clean the DataFrame (impute missing values)
    df = clean_data(df)
    
    # Convert comma-separated string to a list of columns to drop
    columns_to_drop = [col.strip() for col in args.columns.split(",")]
    
    # Drop the specified columns
    valid_columns = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=valid_columns)
    
    # Save the cleaned DataFrame to the specified output path
    df.to_csv(args.output, index=False)
    
    print(f"Cleaned data saved to {args.output}")